#!/usr/bin/env python3
"""
Discard Hold'em bot (protocol 1.0).  Python 3.11, standard library + numpy only.

Design (all code in this file is our own):
  * Vectorised bit-mask hand evaluator (numpy) for 5/6/7-card hands.
  * Pre-flop: embedded table of 3-card starting-hand equities (generated offline
    by Monte Carlo with tools/gen_tables.py) plus a bucket-vs-bucket equity matrix.
  * Opponent range tracking: a weight for every possible opponent holding,
    updated by Bayes' rule after every opponent action and after the exposed
    discard (the discard is evidence about which two cards were kept).
  * Post-flop: exact enumeration on turn and river, Monte Carlo on the flop.
    Every candidate action is scored by expected value against the weighted
    range and a model of how the opponent responds (fold / call / raise).
  * Opponent statistics (frequencies per context, fold-to-bet, bluff share,
    discard habit) adapt the model during the match.
  * Match-level play: once the lead is larger than every blind still to be
    posted, the bot only checks or folds and the match is mathematically won.
"""

import sys
import json
import math
import time
import zlib
import base64
import os

# numpy's maths library would otherwise prepare one worker thread (and memory) per CPU core of
# the host; the bot only does small calculations, so one thread is enough on the 1-vCPU server.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
I64 = np.int64
NB = 40                      # pre-flop strength buckets
N_HANDS3 = 22100             # C(52,3)
DEBUG = False
TRACE = None                 # optional callable(str) used by local test wrappers only
SOLVE = False                # river CFR solver: off until it beats the heuristic in A/B tests
LOCK = True                  # match-lock (see on_new_hand); tests may switch it off


def cid(card):
    """Card string -> id (rank * 4 + suit); the lowest id is the engine's default discard."""
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


CARD_STR = [r + s for r in RANKS for s in SUITS]


# --------------------------------------------------------------- bit tables

def _build_tables():
    n = 1 << 13
    idx = np.arange(n, dtype=I64)
    pop = np.zeros(n, I64)
    for i in range(13):
        pop += (idx >> i) & 1
    tops = []
    rem = idx.copy()
    acc = np.zeros(n, I64)
    for _ in range(5):
        hb = np.zeros(n, I64)
        for i in range(12, -1, -1):
            hit = (hb == 0) & (((rem >> i) & 1) == 1)
            hb[hit] = 1 << i
        acc = acc | hb
        rem = rem ^ hb
        tops.append(acc.copy())
    st = np.zeros(n, I64)
    wheel = (1 << 12) | 0xF
    st[(idx & wheel) == wheel] = 4
    for h in range(4, 13):
        pat = 0x1F << (h - 4)
        st[(idx & pat) == pat] = h + 1
    so = np.zeros(n, I64)
    for r in range(13):
        bit = 1 << r
        so += (((idx & bit) == 0) & (st[idx | bit] > 0) & (st == 0)).astype(I64)
    return pop, tops[0], tops[1], tops[2], tops[4], st, so


POP, TOP1, TOP2, TOP3, TOP5, STR8, SOUT = _build_tables()
_ONE = np.int64(1)
_CAT = [np.int64(c << 26) for c in range(9)]
C2T = np.array([c * (c - 1) // 2 for c in range(52)], dtype=I64)
C3T = np.array([c * (c - 1) * (c - 2) // 6 for c in range(52)], dtype=I64)


def evalhands(cards, aux=False):
    """Strength of each row of `cards` (N x k ids, 5 <= k <= 7) as a comparable int64."""
    cards = np.asarray(cards, dtype=I64)
    bits = np.left_shift(_ONE, cards >> 2)
    sm = np.bitwise_or.reduce(np.left_shift(bits, (cards & 3) << 4), axis=1)
    bt = np.ascontiguousarray(bits.T)
    m1 = bt[0].copy()
    m2 = np.zeros_like(m1)
    m3 = np.zeros_like(m1)
    m4 = np.zeros_like(m1)
    for i in range(1, bt.shape[0]):
        bi = bt[i]
        m4 |= m3 & bi
        m3 |= m2 & bi
        m2 |= m1 & bi
        m1 |= bi
    fm = np.zeros_like(m1)
    for j in range(4):
        lane = (sm >> (16 * j)) & 0x1FFF
        fm = np.where(POP[lane] >= 5, lane, fm)
    v = TOP5[m1]
    v = np.where(m2 != 0, _CAT[1] | (m2 << 13) | TOP3[m1 ^ m2], v)
    p2 = TOP2[m2]
    v = np.where(POP[m2] >= 2, _CAT[2] | (p2 << 13) | TOP1[m1 ^ p2], v)
    t = TOP1[m3]
    has3 = m3 != 0
    v = np.where(has3, _CAT[3] | (t << 13) | TOP2[m1 ^ t], v)
    s = STR8[m1]
    v = np.where(s > 0, _CAT[4] | s, v)
    v = np.where(fm != 0, _CAT[5] | TOP5[fm], v)
    v = np.where(has3 & ((m2 ^ t) != 0), _CAT[6] | (t << 13) | TOP1[m2 ^ t], v)
    v = np.where(m4 != 0, _CAT[7] | (m4 << 13) | TOP1[m1 ^ m4], v)
    sf = STR8[fm]
    v = np.where(sf > 0, _CAT[8] | sf, v)
    if aux:
        return v, m1, sm
    return v


# ------------------------------------------------------- pre-flop table data

# <<PF_BLOB_BEGIN>>
_PF_BLOB = "c-mCF34E00)&7qQSX59@6t$ozD2jCjms0DBii%4`0hJ|$kZmTJ%)VsyeI}VKBm}}{h{*1OEGnxar7kFFg@Ph#KL{^a7qn7T>VLk^HT>V-+w*zeOrHCkGf$rToa?&o!`ogn;u^yyluao6sEAAd_r$fTa#iKmQ@)<UZzoo{qq|!D-6PzWhgP+&>vxZw@ci8)zk4KO%9wsA=K7)7^~|jku_Y4+OpHAldopEPlK;u)l<nclCC0r06Z!mb<%uY+0TZ=$>qKQp*<w#B$HSHSOdOvduH0?gZM)xezv(x}Z;p?uKB^*vzCB|i;>5QsF`~&v&XkQj%F4H=j4WJa;o>)b=Qn=;<2Z6EyE3164t_pddE#5hroY<t4BwI}Jl}4cp6@q@zGt^h-}9T}kK?q?9evv$$Nh2KpUr<ZkDAIgYHGg;RYlk>lU7aIGH%PbL-t$pKlyyfZmzz?xVLH&pUu@bO=dT~p7Ph?v&yh;B?qSj#s|hPFFU6!J1INq8h^Kvz<BO0&;QQ7Yy8I@+~e+Y!{?kb?j3iWcml3flk`1RMS7Y;_ATS|JmzYB<{9{Gu0A+L`OYa*zOSd~sm?0XQ(fZ^jMsBzC+WG4J9Lj{;2zJwz2lBjZ>jfs$Muer0w)Ewy0*IB480lJ;o0Hgsq}XKZ00FWyoFngC*lct&O>&dO;5_(Ib`Qs`5wNN@8KKyoo`%jJPA+A^YWxT?=^m&O;5~U!xQs%uJQ9^Ts#>UPsVk9;i<XHeeQC9TQSe8C)ZP-cyIhPSD*L`cv9_nzX|*WdTO3Pd$_zzJJ_v6JGeamJla3*=r7?3RS7puRwewkn7>3%%@gW<o)pmgd^4oKrPQmx<)ncAmN!G&isjFW{v!FavaR@W^W)}sO1R!B`C$Cd6S1O6MUyOJEn{2l{N!r2A291)(Ih?}Fq?|bnu48K@<u6kPRXt*b<fm2<14$fY<to6qFVnsC3Vkm&zJw5d$s<tPVRBnXZYM%#=Wu5`=@?bhFwwqQ6<(``9&4hwfy|@KvAIR4D;iri%Tvp8C(5B)qQ36mAw<|n!nH58Rm;iSQ}e?Um0uf#5(+4jhOSxS!0zoR$1%tpZIIJPW+uePt;%DYUl6c-T|{dbMg6rdDj#@`<-Qa_Oqtw4ZKmRH&E-Zdq(eIdy(G3Sf}ps_i~TFmwRKK=a=glA2;b4f2h(kepseA@lmDT#1~b16Ys<VMS2$(m*`#GSEhH-;nx~}FKhh0tnv4<*5Q9B^ias}^?NUlTpY30+G>x+kH&Wgb_Xs=UXrX0*M@l$+Wp7PCwBXP-vV!ichhR;eds>B-)iSC<!|FJ<!|G!<aho`pYhl8c6dj;9o|u`pZB4+#2e%-vHP`t{xbds{xbds{xbds{;ICVU#&OAU&Z|n|A&Ql#arTC@s@a3@5Fc?dRx3%-WG3`x5e(i6XR`g@iw@48(h2%E>>A*m33BGXO;C8Cf=3a+T;I!PyBDK{J(EW-SokD-k#nVf4%y{SElYbr$pW1%fAct26yxpc{}PdQ<1vt%o5(B-WY$S`r-_;`eJOgx}s~jy5bD8x?*f~hhII>wOl>Hiryw~McsG*RCV7K<?6o1O5Ud47;i=Ob8$rVb2P5{xg@Fg`A|sj^Wuo!=h3*{=OxJ%CdK2Sx2kwN^j24xR+vJ0d$@$_;gUU1tsal9nb2#(Pfz~z<Pg`I{7*g)agD87W8CXCfzM;Byc7SY2zy}4wPo1;qVVMPPpyCIx3Z7R<`&H@YVz+dTK^RHe#`&Py(a%bC-=Dfo8j|u8TSr4A1r>Y6g#8hS`+rEd15uTt+cfC+X>%JILG{HWl>2{$)(kMtFI`(qWqrtw)}n8&M_C2uy$$wK5O^HFAr@qVoFO{WA#!aW^XlXmxtP>+*pc*Dqc5Z6U|#J*ksEr8}@wJ^wJ9^T`;NA>@!_D_0p-kEst3SRt&7@ZadrFVs0_tmVLhLf=T#PnlGJ-&+hy>KHY6CW_)hT?o8l=7e09HHgY^>!3QsVb|y}|d9Gd)^nO>5*Defk{q&^vVQiH?^X~aPwkkYX@A2a@y~qD4(r#Q^rrl`ruYXGWF}FzjanPxIynF8P?zwl+Sz4<1^=YNv*WPNquh&YoE7zK|D-)}=EBC~|ouGXwD$%}NQLcTtJft<=J!`ys)_C`<T^^cVs`uhE>AgH=(R(@DuJ>|dsdndevvy~TMY}W0rro(Md%+~_&!tnfKLabYKP_hM&(4H=c=!15?(yN><HNhhXJ_K7<W<SPhyEVACw))a95qLKWqV~?5-o`>bzABdrWU6D((sptY&x4}chnsr6T5Mu3)UDr!%neZLtN~Uc8u;A;^OV|zIpq+Z{9q=^X7jucI1Bw`^Jv3Z%uyoNIS=_vU7Atlb^TEd*yBOUU}QRSKjnCV<)tGyy;7gonYtKKX#7&W9Qhvd*bYoc97j>2ia|QknXrA&fDev@pgHCyj|WOZ<qJSn`NCh%Q|m%x3L4-#hnRukbPta*++JeePjpO$J?^(k#>?QU?-^pc9JS!C+Uvcvg{BSJH*8faj`>O><|}T_~C^gUijgKA6|x^oqg1fYBx{x6+g#To!D7+P)NAvDRx%7$<7Gdek&8Q?Jp9t{g&S+;Tm_e!|a?8(mPQI`M?x*Si8v%2-D6n3)3#G7NTt{6{4ME7NT96-xuN8wo>64E81yxOvw3Qv5@nO3L)pGW_DV;$(t6IRhormyDh@9?lvLW^JPM^O0$q`w?#<S-L^9!9K%C9E*#r!5su-Z9cRylsG$lW>O`{;b+UyW*KV>?>bHB+>bG85_1l&@_1j+>wC7hPwdePwwdcLE+Vd@S+Vi}hUHS0T&MP0D+Ii(OJndeK*V3hot4rAj<4-NZnu<y$O&dFHY=1{n(X_FA?(f)Y<}<(Xd8_%A$uE~+(c=G>VEd=6o$}WBx5jrV{i0;=guN5K^Y739&b=<Bdna)3JO41(e#2+&6z*}Cd&68;P5rSPD=9n8g!QVrqXyerc5V6B#a|a+TDhZQNXd|rPK(P@RbEwoJhrv$>tfa}tsGLqT4(+~YsX_x1X*L1HC8)~7?*{$CxW5UA1bhYrSF@tv#Yn(VCPwTIk5fZM=FBF!Qug?!KQUn)=e2?T~pJ&vU{b+vB=hDZZog1+g~0m#%Dl&9iKtg?v?m>9BpQN*4KTKzz44ZMvlQIeDJ~tFMK{pEGn<&Xl7ng{zo&`Q9Z$m?X7v;fz>*0aAPUcM$<_pCzXt=YB2RI?OA%J?GtObGF-XJzTGjfdSG?ZwcPze%?mXbOiShc@EupxvlQPmZQ)9MSJ?+v<C}E7P=oIU(}w3#M$U~UeDTH?Z+!8_7jJxr=T1}!SIH#R!l^~7nEsAwV^uR-&H7B8@Oi6w?G)|#7bV*BmrGPR|1DI`cmB7=t9JHIQ0)wJ=^k~$J?eye!(7*vYj1Z{Xm4E>?d^}{s-V+Ms-Qb+R6)mMUl*%}hLos=s>)SEPXx6_ov=ooutuG*_C)YVh4yf;Nqe}aMtiu(rak<jLY4HsNtLv<MwQgdp-Ng`7c5pSt(&4+>Rzc@YBQ^rK1s-jI>CoJ!G}7*hdROMlf*`o_Nt+<SD#q5SKA%htL1L(Rkc|a^~XX*O|Yt>UU#UXZg8ujE|_*wiE656scI@*shS#Ct(tnFMm04&CtvCWU+M&3>I7fv1Yha|-{Cp_^Lpy_)S8Ah4cDiyPfu=|+;lX1G&^J3jA`%Z-p{RWS>1AL<E@P^wZ7E4r*Tgs6(#iXTBx8CD8tXKW~z%C>+hhhRArz~e+LyqjZiVv2-ULRs3KL`Fc(!ut^K!9Yv1{)D^(knNVS1J-}$K!YJm!&7N`(vf$HEs)xmwLqtmFK9}6`{wNZ0a8#Q-4MqR1us8Fho3Z?2mpW`tqf<31q*mEj^J*OhrbE<)Ls$qapHL5(Sf%Q)kR2{WP)lqv?9koZ*QG4s_s4G<;l}q(exl|vOOZ9<1>+7fh_O_w0w^RUoO9imEQ~-O+?x&1uQ3bO5__6!=vHSQ9&ryBU;2#S$NcB;JR39}+^-+TtOrx$;g;X?ENJUeHR5VpcMN@^K&jr({DlV#ui>l(Hs<@~sE~<(PZ~XDbA8-8e#vgC|_Y^8|Xn-12HL4O%;21x*nom?E)hCwxU_4c+N~Eg9dtFM!di$q{^}3V}bBXizPZ8&FN7YGv{<lz{RHrJD>Ji^vS}DHkw21AtmWk~ytrXjJ=4(n^$BL>{T-Rw-n^^U#sbbZVGO=o}DymeKNR^1^2AITigRJ5?k3%fCzg#Rgz$BI%WEIPK9G@h_ad@a&#c_D3TB$OzZK%Mu`%1;OXIG1D&$Cjkszj=xp->Itw=-?xw^erWThb+VOPR!O<Eq4NXWGPWtL$R8q-%Ii+?Fzl+wfG?ires1)lywz;Y9@&UQ#X=?x+?E@2wFF*E*<LRU*|Td|T5Xe4E@Pe48;%__n%5`1Vq(>Nb~C-L7d+-A-;&-OiY%x?SC}r%<~?1IkZ?s#p1mQ1vQ55vtyyfuVu7YTl~(PdV3r$_Gp;nSz<7EGqus=?|X%*ln8f!P9*H*!`P@&-}*c-z@hQtuDdVPQ9c8TUgRm@{ec!@yuIgk@C*Tos+-yFU<eWy|>CbCv)#x|9aO#!>6f)d)(#Tde>u9M^s?%R{XviOPKaoV>`<GRV*!9T9m13HeXnJVd>H8qt)eQ<z+pRJIa?9v6iX2u#`3KbM;7`7GjN6)>u`{(Q4LC3*A_HS0(mgrL`IxWIklU=2$&W%vaf7@nG?T#fPiDsVXWfDtn^l{+hlOeJko6s~u~r)>iGR^Hn}rjL+fxIzCV2*YT;(uj8|;Zg&zNybc>VzNvb`$Z>y7y^&+J1E1Z=)5=dbV<%Pr*@7LeKFf+V*{m+CyL+bxJ7zj)8c{N$<g}Vimi-m`EBafrHh-nR@;7Il>x!ByYBqaHygr-HW^Xx`_rv$Jn*9~{_P6>g@%_ztMGd~2Jw6-0_Lemb$Bdi@P59!CFW&g#jW6E#u4!;pwOg=TsxPo%FImoYVCx(gyRqxsjXtc|8}?)7nsGJ76~z_3t=p^{D>qhdwLfX^W9nm?;=ah^v3jhVy@$N~i$V{7x&Hyz1Flos&3S)fdguL#*_!t!W{TTmC1x}JULqcF5p!z$)ahp9JI2-YHojw<b*u3mPuhtg%81EF^)SAp+)vEZ=_e|lYf-W4x@3x~|6}(DPpkfav*<H5&*$GPO(m+|NV)2Fb&0y*k_vUfxBh=Tqdw@ItUg%p(miUPd(=Gl*1P&us7{;Bs?!s-J)%P0@cU|Y!~Q}y^hho(Qa@Z+s(zrRi8(E#HENzUYF^!Nw3@ZkLhTi*$8V}skN4N89#=b5k9Sq7E3DP(ibEE4g~zF`*j4vnvHGH@OnuR}LVdBeN`0|ADIaPcA8H;SY91eI9-rOGgC^D8CX4DWYg66Txm0&0Ue(>{W_8D(E$WW5tm+P{OWm>4qwcV`j3`lm?5|LN_$$>PSJbFKd^YvRng;n&^Y~Kp_)_!uQuFvy^Z2f57+0hE+GbUKJ!w~cUF1=H9rCKa%KfUZc8j{?0-L(zT!*^kVz;`a(Wfp6`_(0<wij2ZPc~MnPx_eDCmyT%<d9u`@_<WyGIhFQsCi<jd19z}VyJl|CSUW!Q1ir0o&H|yd#!z&`!>Hj{oUy^n`bu1+v07jrmdRR*51}0ZHcyY%<PyseERU|C(k~4_LtMYoc`vVH|Mx#x@XcQVu!bC=!z38z|Y?-bPRp-v725|=fDmhyXgYzo-Uy7>41esXQ+GDyXYMHht8pYzV*{9>L9v{4uT!N^;7lKH&su4pFn#mp8Hfh_y4ye=pg!t4x*3t7y772l3r0K(RFl^*nwW5Tc~pCm@21^sdDO=DyNRAaMr2t!$v2lo2c-5qZ8;P`iV}WpXem|iB6)QcGb}<>L|LAj-ngsD7ul3f*p3%QPtEdRZYE8)zm9hO}$dp)GHNz%;*4h6%~yi6^$Pijo+FEI*Pucqv$I-ioT+w=qoykzOuK_E9xw|l+L0{=`6aG&Z0}{EZD)`LKRb&R55i)6;qc~F?C55Q<qdRbx8%AjmlSdQNde{%2#($!JCcBS9eXFPG`|ybQb+ZXVG7D7X3wM(O+~H{dH<Py`m1Id+9K`mky(Q=`gyN4x@YNFxcVLb~=TNPT`_cxabrvI)#f);i6Nxh$3!gq05LOjws@YBJRsVmk~wWq9pyLPE(hi;2nPcW;xMube0s_fJt<mx{QvI=6b74ifdts6xUm2>s``Y3rnQ8xTDUa_oUp`PNnnIWpswLR;KENZj#d4Q7)yGsrp}Tk<Ma89r*utk`&!zQ>EzMt=M1aK{`-fMh8e^9j=nbdZI=etKK1n<*Ss!I$R}%^+b&nR(-xdq_6N$CrV#EQ6qg-Z*-89-i@VFdLLFw=?yYV>CLgyiRv;cUE1oj8fmNkR%xr>oYGdCJyKT3Oj1^-)ks<Ow@O+4=9IG9>{-(wU3JVPU4^GQQo0IHbtIi51$bJy6yQk(1$ekx3b4s01=!t9N2<%HXlbh6R%xoOc4?|9ZfUB`UTLZxekm$*jTBXHs}$8%yA)O4->$xtqUw>af9WZ6jr3G+tMt@XyY!S1VRWYSR1c$LqzqkEQiiuwOBudokuqH8kTSf^EoIp3r8CuKbd31u-Ra_^c$@gBtzCT7F;jeW@@(<Zn{(8g@3qb>^rw0=-lpDcYgcb}%>1&@pX$vw=PXL<`-oJB>idXPhwA%?REO&Oh+LFhU|(Rr-^6vlsr!^=rI@L7)YM7&|J>^}l}?((=X<?x+4#(Fe16OJ^wiDe*h3X*Gq$R1PkCoiXVFe`|Ehtd151yFR+V)Yac`%2U@7;GhDLk2$6fAmmwThVKbJ;L*f5iE@T|64dfBja<*|y7Cx1NoNYz?1|DSO?Z4cQ-nMRpxiFD=1lUX}bHKB~Po%#E$*%GIOS!0zoR(BdP57}5dCA_Mv&4hI^w^U<K*4%8zimaD9u@6iorsgTlQ~Fs-s|S}4E?;2(uWf*Nfca;~C<onkX6}RhIzIg@gUj(*U>{({=jZ%7K4<2(CO<H8l$h|r3m?3GHgb$|;IlP(Mdfxgc2U*MR_qPy5(jpHy^9-L<2kh!>oDJKzNzG<5|8!wR!60y@=oV@j`r&I>f1a=omX3~wzzyNd|%qXv|riUk@v&bllQ~-PG@^HzPEX<w&3f^`{8?KtGS`W$a%NfW90n36<@sZ#oJ}%T;apl-0*94Lk$+P{MCkSv|r%HraAxR!6tbp2e2OglS0^a*0t70D<7@A)jr66)O6HT>RIQywdU5Ez3%HgZkyXSEilTz&$Z8G3w;wD<{jpJe%5t)e`0RU`x8^@xwVFvy>7RSm}!B1E@Et<VP0aMpVg)9I^#RmT5mPJW00K~qKF}i7@~+FikL2KMfN35?C<ud2P<}d<iQedvkz;ibp){`fomh!oncE98}7Q>#fKiABG18^gEgDH`@9wQ3j3hiMqf`yPsgs{z)+{R)7u{26~4xIjqhm86mj~U{<itU^Kr!X$j1@8DIZ7dpxT}eVs`~Qy~MWX<A^;PbNY#GoBzq&;l_8~?doBCXOU-<@tym;#1c&`(e1`}?g|r2G_jw|J<;7<qo%5dmzAp1@AXcaq+Wl^rqA>@pWm|WDOa!duTrmWE?38=&Fc80q0S=p{J>K6{AjQ4(cj#ozqvQsd#pk|xz?<ne8{GrjGEN>=4y3*FPl2wmiTzGdVfNhdViEjy?;trYxFm3^fzntH*2SaOHAs$(rWeIe{Jf$Q4aN9n@I@JQY{3y*)9aQ)F}iwGuJ#t7%;e87%;#r46s%U1GXmRLx1B#f8#@c<3oSrvo(3QSv~f9t9tA_hkER&Q$4oArykpG782ZS6%s6Q2no8lg#@S83JI=ky{SZ4;HVT9v{wrYuC@pZzO)Mq%nkCTzwxEN@uk1<rN8l|zwtFUthK6_2HDk1>s;!k>pbeEQGWH(H$nAMLyZvOuQnmV1#Th2zdS;O$pIn4Ng*M^^RphU6eb)s2@`Iu5hl28!i0S;VZt!4FriDEV(4#T=x<`^Z(`_gV(4#T=x<`WwB7Ae&lGvoGyA;inMR*_W?)D?vn#BgF-6ofOPoT6s7J`~Q2`muJ|Tl6C}g-cB4n^cg$!-;qZVPq!5U#hg<aUt(;;lgf0uy`*Z71DPQS3>levnezlo*4iKV}ZrN4=#zlo*4iKV}Z{bcTM^M0H6ul9encX_tUvoqUgw%_{ft<Ro2``p>h9nBrXXAhq}Y5t`7gXRvJ`*i2ioxjfgb?%<dJ)M7k_Rr5cpLafgQRhXSAcNR{fgL0`!Seik%LYn-7593<2cZS*f3KI$r`PFxdY!K4ce<Y6M|(jFFaxvzGmeJ92cZUt0&2kiM?-WvJx!<6({wsLO_y_@F6Ta7zSBT~XaPGw4X^{$06T06@ImMS(tsWy4d?;;+Y)p(y-R1)yL2|aOJ~!&boB>D_X|PjYS!uMpN;MZJ-`sq0}KH@z!1;_3^_9gJ_toXAW#GZ0!2U|Pz3frGe;-WqjWMoN+;8!bTU0kC)1;Jafi|MLK3<dKe`w{y4YoOJtzW}fFfWCC<2y%B47z90+w9a3O)!;Kqk-xWCBe<CeQ?A0!?85D_iMYdeP`yJNTepq;u&-I+tFgbLmC8_Bx~6g(!3_{&X$=bS?gLE&g5FKoc+pGyzjU6EFod0aHK|Fa<OLQ=Xp%J_uDnEKmi+0#!gPPzA&SRX{9I1@?b_7M)7Z(W&$tol4Kqsq`G3O3%@$^c<Z^&(WpBjV>3m(50J<E*G-UrNq&t?M9aiSw5Kys(>w^3fS^d0b4*7umw~BTR;`C1yliB+UA1~LKlz>bOFgg7my5eF_0_>J_ubvGSCI~Z<`M~aDfh7paU1^zy&&Rfeu`t0~hGPMKtj<3kX9r@xu!ULp1Tf77&JL;)y1HPZQW8lo7(5&>8%E%XR{7Koy-nx=#UZgfO6m&J#P$I!&x9(`jO-d9+t&3GN7WKogxxo}Q{x$wL*Ojt~Y^&^h8rmCg}6Z8}AyD|L!EQl(SG&OBD=48e-fM`wtg28!sk@^h(9E5l4WtynFfj}V6L*ZHBJMdye7-|kTN>-6w}NvDT?7M&gz*mZjNIgbrGJA7c$*#Qrskj@T28|a}^%&Ib-V!D`hig~g|r<fuuC?tfT%XMz>Saoi=)2VaAZ62K)Tt1x|I?OsXc&s`#-09S*;Wm#>4X!*E=*-Yz)|tU$)tTW=r_Ky`3XMPwoo=qE)amA;DxGfLu<CSkfnBGYH6GAN2tzmPyl|^s=Y>*_&I^0pIxkEM=)7PH>9la2Ri}kp?K&-#dURUY>(*&uT0o}-Tc}H$&I;FAbym35uCqd^M`r~hgi1Oq5Fu0omvrh0S#;{zXxFJ{np3BqNnV|LdiX&lAq-usb3zY~&Iy~mIwuUO)j45TQ0Ij9u+9lbV>%@acj=VS!=qEeCa+EjgKBk3*cH?%p*^fq!qM0#b9F`-?$Q~dhev0GO<tW5h!i^MjL;s|8G%Tl6ZoXl(BJJJ74S)?p@dtfp@v$WhL!|$8oDzKItgJw3F(h61^sdBv(g{U9nv3@=1YG(UC<wUI;B6H&kLV^n>Vw7QNpKNpB-MnDB;tj`M(x0O8B&=Q~2b3eovFWpA14NeLoq5Qu=-}2&MG>WDrX2Y1-3t)N#~tUlrGVRcDmmTZ!$cJh$SW;(LmB`*u{`Q_SbxzI_fp^BbS{IUE(k%-GpgWi{BRRlihDEty)fv8I=0R{5;*-H}hLrj~GTW6i8`?(L4e?BgDHxyN1Zz3jWDa$gmeuFl!8Z>%okKj&9pQT2Z5`=v9jU2RWQK2_P*eVgke%SV=-sq?GfFJ*0}^{GnM`no@|u(mVxd4x4qS!0zoR$2QzGPL5D30r8s!-|cwp6SH?=D5_0J#HRrij~AlDy(xX%y6%^_i&U|l~w)hdBxLF(^2z9{p03X2|g9p|CHl%wY{tgpP%#V_<T{{nZyS#eDJ~tub+(^uXyn3Oy(+{uEsvCI@N~twQqJ|Ew1Z*Schkg54)^-t$BXw{L&|Ehpqpv{CB0-)#!TB@}gyr_X)4bZn8gI`(^C}=LF}mmdmQ=m*V?`?cbI7dh>qx?#cV%`*7Y5-(xLrG~kOjzIc0$oQ<wMM$RX^4;wkZti|_@hAPWCE7nnCc3^*TuJB+9uiK9mdj|%vvB5t?up8_nZ7WSHO=sF)wf)KbC-Zvubk`p19_w9!YX4)-$DDTsDgs}+zjVJHxhK4#c0+B^>>KPWO~jmO|C5=R_3k}ZV(tn&<|O8>ygxB-M>f<FQ#3o=MhsEJ5Je18#1M6t@f{TbVu&Ip+_uTS-;MQlF7;slaZT}IBWte-VyF1agV^_>ZBcA^bbA8(-j#5@R`Xg-gLl8Di@l3I>+9*`zd>_K;Ap_=wz_wQ76kkF`uJL+_eY-&J{?>azc9HvygJ;k^ZR@pu?^lXc4D*nIATx9#}T_T)W=6`OFoX+b@A0<V*7R8HTQetI}<KqiOw3|+0%E5@tsEl#1h?NeCPd9Vu>d9uDQj&tNhr1eVHIO)BivSI~+JQf^~#Wk7C*A;w1LZc)twxY4Xhs*2llgpLV95C4nP>zV5#6_Hf^D)ED)|B3q(Y*Ir#4j82LE&Hp$5p2VTVqv1!xqchtxEzy?fkD32umdBRIa!dN;<H=Eyk0(caKAs%0$knyv2<GF-u_qr-j?wvea{QQ&Cr55cW?>)W8Fu+gjAu9!Xg8jrZ<rip2^!BZCA!CWhC_+b#xrcs{AfJGe=_9AEIfg9T<2B@&+e@h>h1R3Q!L!u=g?=cj?eoXzf=mhdRc^9!^}dzvKpb^?#R><;oq!s;or+X-2>~m2i9@#W#1K5!l$k_;nQs{;nTh<p<vD?6m+?Sf;&_1mkI};suT`>WDyR29?=?D#~N728d%5L=aI1{;m{n5aHxkvIP{7~ICRV;G`zzqG(6KOG`!R+H2k7IRw6w7Pr2~0tV(#;Q6oI;Ov(qW!w0Ox2du*ftiz`>xz;SaIcyc)G`fU0Pk4nlU)Bn5o~{-uo@x^+ZgvS3ulETR*A!6kSj+rU;o`q5g^Mp*go`G-aB+fDxcEkce8D<=!8&}wI()%8e8D<=-)I<V6Rx~kz?JDP;R<*Hu2cksEBAzjE9<O6N3%odxWXfJbo+&l0|P?GKSYF%MYC6$gpYqR3m^Abg^!>dSoft{__(1~_!w?e3|L1DSVs(4M+{g;3|L1DSVv5_E#VS=?Dq&idisPPM+3r-1wrA*{ZZk^g-PMZez#C^sYfU|#V3@!CMc9FFQDYMs8DiyLMYj<^R*h`WEZ<|vbBJdR=04nk54%HbWk|CIxL*LYp!C!I%2`P)&f!z3)T?});%2*QW6W+5qsC%U4G%jk$`ZbZ&*07B`TblQoxBr3E{-{jBw&V1)R9bFSN`Ag_aM5gqEj9gqEjAg_etxLd$*`q2-$ew9G9@JB62h-NMVLPk4ECt?=@1e&OY#Vc}&<RCu{OCcMlnR1UC?9AF(ez&dh(b>sl+$N|=o1FRzlSVxY`!o3UjE_kftv5wy_{Qbfi^JmOIY0*iGj?6zYfAFHgi*|MH>YTc8>cXn$s-By-Xx<{{0_TEVi+3%qf4=_tjY~H!{q*@ypWnS~_p+`_x-J1Nr4EleK*tj*ke~Y;ASd{_+XrR}K~aaheV`%u2O5HZAR)hlg#7-p4+I5AK~QjXcLdB7l7g}zDe7={1oQ*vKtFH}^aJNWKJJ5j+z0u%|9J!?1y4az@DwBkPj{xkOd%@h3!;L)AS&u`X9~0f&p<oy473B!Ks)dZq+=bVV;!Vp9i;o&KtvD~Tm@0VRS*?i1yRA(FY3WeAuFg1vVzJWE2s>zq7J{P2i?Fe&<)%I-M}r-4cr3Vz%7ssKadSSkj-l#q0kj%d)PoikQICdS;1G36?_F*!Phkfd_C3zW(r|JYY-N+24O*K5EirsVNr+2T0k@K>D2;0fo9;7lpB~Sd;-nDCy)$(kPLs241cLM5Kt%!lHnh217X2g5Eh&TVZm7t7MulP!C4R%oGqFSW(sLRagY`i2Wh21!Av17C=Sx14vS`kUf>Yu1rC8;;1K8q4uM|a5a<OCfnMMc$o0K}d_vpS0@{LHrx?g5v<10Z4CE8q-ZdAb1#dxG@D`*6Z$Vn{w!DD1AT4+c(t@}BI>AgKF6a*8wiXZ<bO&)kcM$jKpb!^y2XRq{{W?J_@CLLpP%|vN0j<EBDFwU%t-u@53cUGG0dGL6J_h0mbwR3j1M!5qAXU&nJfSW~HQGQtp>Ad&hzst5xZo~`3+{rr;4X*@?t-}BE{F^6zFENC+!8QT$P4O&yr4eF3+jWspgzb8>Vv$XKFAB|gS@E2+!7Fz3&i9CF}XlYE)bIo#N+}oxj;-V5R;26<RJ@rjug<BEaV{zdB{Q@vXJM~0{W7LJl+}Lt`J!0dqS7;bD!e`5`(xpxt~!A5(|AnQk}&&*61YuX_Zdm8*5(n=^V}-Au>p-6TPECC;GFiKxCmWh^aI9Osme|ecd{NpI@yL_)M!#;C<bnM|A#XMaZo4H!DJBuvsVlbhS?U-&jFrp)bg&vv-A6XYZ@+I(z@@(aHO9vrgU>R-L@Bw(I2ma~?N!?#4q1t#daXLTHdx6M&%=ngA>`YXUINstLf~93ZsN7X;Lq`w5%Q++LT?+<Uw_b3a_G6Zd7+I&nW?(}~;b(usSIS10a=^Ej#VHl9LiowvO%owxURb>4p1KvYczauu2kd|IW+KwrBi11+vK1*`_Cg}xx&s|BRfS$n<v|DLt)3h1o;c0?!b8ww}wGwnKQuXq37llEN!owVPMgxhq^zM*i=CPIk)f6v)>1$53PLWm7!Yhut*qlv*^oSGOU3KN53uO<d#gCMrh7evz;yTPk7cGjme_9+3Kv3G`a#%_t~jJ+<d6ZZEmov<6ctp(iH3Hy|QPS`s`I$^g&b;4d3ziY0}*WbHzz9v%0t@HIM0iCaj6msi)O{9<;?A9cqw^NgZ|F|?s7+G6hz-~<vz7J`VFgyx!3w=Q@ovriu85VNsY#obCDIk~5)_W2<TaPZBt$)nuWZlQFlXXczC+qgGPS&xAPS(MwPS$%8I$4h{oUDJ$<DSmdef&CCmjraKZV&5RO$H&j&eeMoI#-V_oU4B{kW&+e;=+XCzdlVEX8JW@I2_P~p(CUTLpG`j!$0G17BCzH7y5#n(yPB;D7|`8L9Y&8B)vLyq4es!MbfLg7E7;gTq?b~dzmnN?*d`??-w2^;J7e+@FHRO)P=(Ed5eVMyB2?1z;R*t?q%K?dLA+f$&VC}T+c%$A-SH1OhWQc3rMc#A(MB;e_j7|ZMSf3x2&l6t_t&1MXDYweXukU@Z^8;IT2`ad5n9JDn7TkR-32SVCk9_c5I)ezvXDz(XtKp*X-@)cJut$zWndp+mQdAd-G$x{re1`{+11f&uez>_4eOfJ>G`Z+iq}Sb*>IC_K59P>&}Xu6^*VPuG6beuWs_L@T{`0vQNxDV%u54T4Vk`Yfau&cGf0le~GciYNHXe!^Ik_to;(ZqiSyr_LQ~8f&Jpx=fM_u{wIJPtg+SnuHttUKRUWP?ykPO`bzK3o>E(>?d;$lf6AG1_HI0wU&rUi{5n2YdP{BioE=O#@#)=ImN{tTu+{u%<ml?a2QPf^!lx{Au6dRP>u2fWz?M7Sabx#+Kl5W*|Ev%;+j6*ieEInDx%O?gd8T=$zMfL|RO?jhkN!W`_Hy-dJs$ownD(Z<`=-sdj4#J`u6>>f-@cxyR(yZ-_j2L;csT9Fci*%?4qv?S#T#F|@x>coyzvd>##$#husYkj4y?lYtPg9i{UC^q3T}*G^MikkVIv$DI<{ABue#lRh0|eiSQhxc^6s|pwl55<@W0}I#eIJCX!zsWkMlaNIg-u)rE_PE$oms>yW3$QW`S?FotTAzSKP#$AN{zNm?_b$pP0F`TG~b!-*KUX7@~+FiWs7ZA&MBHh-qm%$@wP__O!d$hb6rm{Mc3g<`A|bd?beL54+>og~@aZ>+L<=Gt@fNdUb7YUz@$n{;0pbwvV%qvs-vh=wr{vo~L3zL{AQ!99SFg9={}bNpMc)-t-@$e~4bw+1oqRO6=9OZFXWG_4jcS+b#UDhuEiLCkKdKn~x)QPUa6$Vz22OGM8AQi6xp?qKPG%SfbY&-`PD*EYZXcnQQVb31FuL&In_bp@C8CrO<^j!{V1Gv0vkFr?BIxu34-;J35CYLz9C;JwrY3ghmG+@IBzWGI~aIbM5BZpQ6#2$#3#+NCuK;MbC;pmzt2Oj8;b5vUk)!mwYa{FSj5!J~cj7yfhga>LJHFp$B~AxH2D4j-R3?KRGt!<H_+{KAs$H+2@kv*q4tdNAc1R7LtQ3<RHtH#xtA|B?nnH7|#$$K4(0`gcLc*vd?&i1vzqjuyC9Irw|qhj*4QY$mKEY%Sf*zc1x^(3M+}9l*X2%7Sv&fGw;-6pVU3nh#d%>6@JkBp!Z1R`e>pyQEQIRh{b}j;C0Dv$uq-ehTVy(RB@y@a%tM1+7j6knO8rkZe(I)qP6~+`nythr6xC=+xTnx*Yvs<4ul@`lIut$QA;j!JQgI^b@@Exawm!-<hnGyB|@%w^&=DHYR%^%*W`v@)8tzB!u`w1MK*Ggja+2A&UmJ7Npg{mTx25`*;<WfdZwOSWFy!8%TKTZSENeV|6P?BA`y77R2<Rb(r0*q&n>S07GeBrc47SV8ZpHRyO?5r>}Z*|qTMX6=<U}%c!7KH0{43RZ?y``cesS*D?GyT@isBW4K6W8hgXa-F}t%uoN;=!IAfJvobgLcYw!YV@B(Y_0&Bm-Y&F8{t`1@L%^qR)9=|YqZ;hCv#v$g|=Mi)KPe9DkyYY7w;*Ptk#T}(KaYxE2?kLO12VTGjUcd)lzz1Hyrz~^0TG+hJCTuQs3!DF3D{THXC~Tf(5rcGbh(X?Qi$Ol~i$P|E#31{ojV~96%rl8Yrdq`zy<Fmuv{xJw$jKL8z!zS?7hb>@UceV#z&DV)&>;-I!YK^?$}0?B;THxU4GV*tBf{Vb4l&8Q1txjcCnotIC??q$5tIBeCMKCXYkQTr#9<Ma?6!+bUU7>{KCTs)Wc}ijmNvz}3&g++#J~&0zzf8{3&g++#I&@X?h)4Z_6cj-YlXFQLc-b~qQctlabfMfX<_Z3JYtk$pBQC>UyRZm5~Cc6iBa5fF-kfmM!BXl|6K;BwAsZeeVpQyk3Hg)lLO+EOM>E*KSaeTL*^<LULY1;AQoOA7G5A0ULY1;AQoOAcF5exL1F6X0;Zl36{bdG!qh-gm^vXPOueICn7SY*OkENXvz!qYvkZ)iSuTu;SuRhCS>8^GS-NJ$ETeN`mg1#DJ>r%JeBzePwc-|&U)*w5RNPWo;Fjl-;+F9#amxn_l>=TN2fRQIc!3=70y*G?$^yHP1708pyg-f*7M>Lrc3vM9cFu?iJG&)?omDAer@w%mgX)Bx&(sS$&utWT{uC0!jEah3E{};}dL_j${ZnF?lhR_C1$APWck0D34>gKm*1hncR~(b5701Ma;+QkT;+W!yIA%+MV@4*#F?XfJF~6q8G50T5E_i`l@B+Eu1#-a)<boHr6j+8_@B+Eu1#-a)<hp-(x4eZGg%+K&{G8?IE;)C}7cYGA!X-;CS<>yrZZA$=I(g}s7stFfaM{3RLtY&6;tfl0Slac)t}mXy?EGcNUp)Ted&}Ni_V&wfzkJb)7ri)c#kdvr7wj*<Fgktw*9D`TIAwSYc7bi+oJ0Vg5$kaJNCaRII0FWOGhmN>2CIm9di!A=xChpOd*;XB8L<z{1p9FMm>+{N;0722Zh$f12H1l8utlT6CSsspVz3V!1pB~2un!zGF$>R#g<vdL2*!eiIDJga!Vqu(3;_qg5O4qt0SCYi2Mrbx6TuFw!w#&MWndw=2o{2iU?I2&7J`d<H^MVwBbW>}g2`Ybm<%@J^wGN!MgZ$!1h5`P0PA4{upUML>tO@@U<3SM1N>kE{9pt80y)?SPJ)f#B-jW}f{oxL*a%MAHw~T<E5UHE5)20`!Emq=3<oQ5`q(!O1_0Ax05BZ}0MlUrFdYT}(_sKG9rPbzu!fik^v56c#~<{^AN0q+r43eso8B#O6RZR`!AfuwtOPf~N^sNMS@4Y53Fd>HU_RIh=7XJJKG+H7gPk~i%$)_|gWVuL*bU-?-5@^L4dR2{AU@a);)C6wJ#nBtaiBeMpgnP*J#nD@T7xabP($XzPH+_L1V_P6a1`tWN5M{T6zl{?!A@}0HJ$K`SPDjjrC>x@3PyybU_@97MueqcL|BT`$2FZGJQzK?fYBg47!AUM(I7k+Z6JIKo)Jca@L)9PPCV#NJm`L<!4hIB(0zl!5@IUQop{iFpTQEa6kG*M!BwynTm?(PRj?FX1xvwIuoPScOTkseOW_%@6-)_R!IZESObJ`Tl(1E0fvsRl*b1hEtvG!YF9p%TVh|lH2GPM{5FIQA(ZOPW0gFL&uoy%Ki$Qd-7&IpjXigr`oIIfUbp|_#v9=T#3p6JWXx?hDgBa`n<**f;1zW*cuoavITfted6`TcI!C9~soCRCKS+EtHweAIYMyv&c!dfsWtObL@S}-WAwWYvXFet1AgTh)cD6GZlW8Dj|3>PfJ1<P>3GF-3>7c9dC%W%OmT(ArmEW<@M@{x^vmlT+bY~&*w`N&2-vXSq-0&|g#d}JfvqtC)wVlOe*iE|J?TU;kt47So#ZAAqvCgy^LG$-3&*OY9ZMN_g3c}~)dY@bClvJD1HY1+2htZ7@i1~wCO!8V$YHM%q(Yw~J3_J~c>u|}7sV@=*)Vw#PyB39FEj1{pOyrwB!y-icNIv1=a=7Lo;7yHqnx!9Fn&Be|RYASZHMpLmL9h!<=>D5&1>|j|&GqHm;nu-1B&`b;uu^X(U>D(Pvn$A6C)pYI`ho*B2Jg}RX3--`FY_47Nu)ZG6!+!K@9`<-x)3DhVO~dBeH4W?Q(KPHwzouc2hXXmy!tfN!X%>d3SWdGrJjHVGoThgDESlOacW7#NpI1}6tRI#WbHNgtgWc}d9BhG4bFhU0&B4wu%)zEaH3b{t&=l-;x29kVe42tS3}^~=eqjnWCEC)a8Q2JiW?)2!?KA@;LTslQ7!hJS`0m{T-)Wjx;nXy*y;jq_QH5#V{2**6=7J40|GK(X^RGw!ntyc*YyS0AO!Kd`am~NxWHkNi?bY<_>RL^|9`$Sb)h(>)*HbY~zt+Yz{hE^*GFP)NBE@=|eGw_v)9i~#v7TmMM2hv`Jx%$Zc5BL)^lHj?m0wf76@@9^{;;Ne7bamnF&C&mx`6tcdtF(Wd;JvE+-pNpbFb%8ntQcnHTT+=)6^?jn0mbv($wq9!qn@hsHR>Ul9dJS)6}aitEtz%JZEU;l`PD>$RPI9%!>?SzsdspY34-+u^+6X>7U7`>E9^<P5&wj)4!KOn*Lo_nEw45*YxjrO4GmkEbJ%d0^K$5Iug;m%N*Cd>$;@oUG9XxfbN=i%`42iTI)6Mn%tmi*MX3xT}L9CcA4Xvc3qd$w9B2?Qs6*MyXF<9U9I(+c1>=$f4OE|WD*N%)<q_<pk`fU5({e9HLo!1YOU9-i%eoccu-TpKu}XbQ(-FjWkgfKTVk3Dmc%s`T$0jM@Nh;`!B6U7K`|F>qw~=@%XL2b;su?Lx)shxV+!Y^A%*i%*TVVe_=`Fpz5TMzN8?tA2fGz`@SNqB6u3}4*sZ{WV+uStq`-q+3p{xIMe*R<FN+7qt$6fVJzuxN^N~?(sORfec)l@(=NnRZzOIGmBcs?*&qu~bpB>{H<2zi#b+~48b&Cy)+BVk=s~lGOMks0<R>|i#LfhPY<~KfXbH8l;w-dY8HPV9}al~y4s}@#0={0-CS;tuqB#t;1R&no1?>H;>4kUgGa*w;*<1Y7p3U=B*b72eJZ9eR9;a~jLy3TQpvW&8H`iA;0cU<l$4xSgh#&eD5q58G?`>b{5@3U5%zt7r3^%p1C8Zqa%I*pj2zG5ThydY~ACp&CCUDzP^`95sCZ%hE|7CbqMt#>*dH&@+Ub(Zg1?`5{jY&`>u{O`Knb)6BZiyZVF^u5=--g$EsK4<wZv*FV-@U9D=GxF>Byx06$79YIu!3!U}&M<P+MezA7JJmYGiLG>9?!#X3J`%*<49<yQgQ5?|vEi<T_R&?Nt48@&c?R1C+xi4fG3JTK!t=t*YnRtPmI%asL0>T2HrzG33g1z_!8UyR1bVsfJ(l;w_pyX8h;O#7e>1*#<BK=Gcpo!z&I{v<H@^Lw*VwkZvAdlcJlH=yzXmX0U{?hDBs3_Fg<?z6*e}jH7s_Y9=M~p{>wK%lf1Upl*Cnooa3tL9YxZ3he>7$ZSOO=fzfA6q?v4(f{fl#Y4Ke#Y^R2{K^8Um$<o$`cEN%%9b8>ojl$fEj2hSjeC}N1R7~gT7zrpy9NSGL+h#`uY!82;yJ!`QpzWxF1!rEgYY+`U=4C@{(N@71Id>O1PwKI!tu07Si)V|cd&2RQ^c5ZeK3f&%h$o-J}_2}c#Mt`IKk;IwF3&R(Nk7pKTevSSbHD`CHyT!Z3-=4p@cB!4%ZTUE22j%04eLWvX>?4T_!^9rX#}RAJc8e4H_Wa}Xh$VWP@ttP>Amcl454~=D=i^agi9T+8=b{X;L=$^_-rztR|F09_aHx9>+Z1{zj`fQ4OJS>%k7Q)Sa@hA-{?{Pi&+Tlut_hnJzCHYl_ZP1v_J`O{wLjG!h<_O$5gZX5kX)VI9@-x2n%$9h#+<Qxa*OL1rxvGXH!R5YO!rKmH|_hTuj{_9yJ+dGd^|ZU`FL_1$j6goKt7%vU9-*@Ir14&<e1&iGfj^3rhQ#Uj*FJATr|sghTFpy;~D-CJ77G+m+=9{GptUMgDggld=_$$g&brd$I3+ug0&HBNu(%&og8zeuu+NDG&V7j$Y4D(-_>IsnXWnP-dw5)JKT6xi(#>=q8Ha*TstTJNPI$ILg0qfeaV}{H-}5oi&LAUo1<$oebO^xGh(~y2iH$dPEPi3IFjp^?w8IqJ=V0dZfD(}THkI?<x;tSymVFU;#zXe$>$;04XK;M<SI#Tj*@FlW=4!$yYhL+)xV)%np~ND9&-JuHI*aRKVI6rj9g@!V?5I%adMH3Tx25`*>)MvG`OBzWXl-O^jH(Q$VRTs%Px<65y$R{UzWnoPX^N1^XYFg*u?b2dhEPx-+F9Ic4q_jX71%?Y+cjDR_wv%hgz{k2}}I+;OW7?q>d$D3%?fbk?Bk?jxLUF%HCd=OXL#g)xB0fEIBOc%z5g^q{gJqZ%E`GPCuL;)qH+4|0_(Umakg6H*{}^wcgeGaO1;`x36B5I6X+tzocFZle0%=ag>~!vbh8~&#N1jB&RbsCPmKk8y-%Rb5uSbIZgR|<c#I>k@NP|x2zy1S;<LOa*~ytWF;qA$w^jnl692vjORC-jA#6+C1yP1U9IH2WyJ}G<=R{$9&NFSZQlqDs}$dEbL%r4%ja$GxJ`U%_J}Y4?G)>d^oVs2Bo<bQcgI=9yFUeW502#?9Lv3*g6Fuzi$i_l#q)yV#m`(~-!`Av_kobu_o4bx7V+=p4)O0b9`WzRNv*-LtiiFY!Lh7eoOC+Gf7g1&e~bL$zq*L{ucu2ae7;XCJSHF(J~=8Dey{oFD)I1THu3PgF7fa|pLqDQtbE{DeBfAo;8=X%SbRRqF0_m1R(Zs8r})Kl^TOh}KwLaG#3?qu+$T1EBq%nX6A>Ff92XmB+eTN3j|bbt$Gu$Q<K?yDV_#5w+`n1Aa4f!XEWU6ozHltQa4f$4o9kTSvsVgycAZ~*772^b9*v35zD$bGw!6j38$4p=Ujt(0T@kVJptx9hNm{HtboTTb@$!7Dc=-~Sc)8gpUbY0p%e$lE<-s!)1IH2r$6n$RGZO>H5(CE)1IH3Gc*d!I@s_#3TepYATaQP@TW2Q4TZ=N{t=(zyR?k|obN_(Y`B+HoJTNA9E=r1>eHpRy&aBw^?fFaX;^)my@$*A&@pGeJ{Cr_p{QPTF{M;=rem*`=v2ZN0a4fNKEU|Dbv2ZN0a4fNKEV0Mu-5wS{{UIiP`Z6wlTAdU>?MREC7T1fP7UaZF-#3Y$+QMS#?lH0SLvgWmzm!<|k&IY+SWYbceu1U0YZ6OewDcFRc>1SW@$`tGczSzCJnf8$rx&Nh(>>GT>96a=(<>J#2OLWdIF=l6EIHs<a=@`hjy%Vb1CAvJ97~Rsi>``_haQQGhwe*?hZd*ALw(ZXp~3ayp(8o*&|^*Fp|_jGL$wjHbx}fW?MjKQThn6eL`H1=UA@@4YffyPY7$#t)gre3$4hxmH28W#Kzx03SbV)XD!!f(6JJkGim&^n#n(IQ#Mh~u_<Hj)<$`0$1;>&LjwKfyOD;H;TyQM8;8=3OvE+hd$+dZzB`&@>mK5J~ro}h6*NJals~6vR3Vf5uiEqwt7T<i;BEGq+RebYBT&#UrN~|47i?zSWh_&;4ZLoIV0&DMV5Np5OEY_abD%O6ez}mO3K0PSjel0BCUK|x~=Mv)WVM+1!n3Q-s|7-?t@2V4TcW)4HKinwZzGa1S!m;FpW624}k`s<4Cmc&oIF_7nEIHv=a>B9Xgk#Bh%Zj#@Z7V;1@#7ayUwQh<c`wg<dGD&dt4ddtt{Aaq#F`l^W~{h!&6R7yuY_MYwCd2R7gxTxa`EcLt3O-u*^1L%J?+)2S6;oc?A5YYN3R;aYVMl3YgVsZz4GTZKd-rI<xMMLZJmzC_+aT1r(k|=bHm{9_!}X(TTIUB_>B;(3=hM~@Gy+b?=UjI{}hDD;d7WAK0lCvyT$0RJ&exj_&@>{hHqhE_!bt1Z(-oI21AS0Vc=qep<#4*9Y%-OjsJ$O4_1e}#q6j6m>m@WvvWFrs2<jZS7BXv71o7UVO@9?#$_GGWgW(49mYMwU}l&deuvrNcbFZ1huPux_nP5uF+A!3hDRO1@TdbAp40Jr&9E%|3CqHtuq^xu%fg?qEc^+>;s?Xx2gBkA!yYpjSuEeb8HR`FVR(2RhKJ{2cz7O$hv&0xaJQHq)d16@8en=<15A%<fay6MXWL*^c=DA3Pr|D3B&-Tg!m98jtO`%UsQAOE_`|3cgNeoVFe?5qD*l6K!1VAvOb_3~^zc1Q58uP|@I6cq-w&M)cZ=~+6EOZIE-^l80>(#8!1$;M7@yPe(Afry*23N5Ls%3(ghk;)SQI{lMd3qO6h1T<v=#;y>%*Xf3<ehK!=S{$pvMgc7V96M2jj#0Fh0Bw<HP$fKD-a(!}~Bkybt5U`)|*OyT$yd445C40rR6WV185v%#X@|`B51#Kd0lj=fj%t9;^xP!J6<MtO@VIn(!X13Gcz0@E)uQ@4=X}4CWR4!<YvQ<`w(Hm_~d`><?oS4`UL)auLiA|HJ(7Kg<vR!~F0+%n$#={QTed3;YlB!~Yj8g}bEyP#-7&)CURx^??FFeV_nPA1DCS2MPf7fdb%ke9=-^5`Kdv;Wt<keuE|9H&_yWgC*fNSQ37NCE+(%5`Kdr$pb^qF&I}W0EQ$F3`rgsa+krlQUNez#$a5jfX&NL0O$Y|06G8#fDS+bpaW0<=l~P|IsgTL4nP5*15g0yfPcIMcS{MNN>Bo*5|jX{1SNnfK?$HrPy(nDlmMy(C4eeH32-|8$4jsxJO(SmW3VDT1}nm2up&I>DexGq2#>*v@EEKJkHL!Y7>r0h7?FH1BKcrM^1+DYgAvIGBa#nB9Az-A)BubaGnf`7fF3{zpa)O_=mC@fdcZiprzaYxck}>C06l;bKo6h<&;t(@^uX<_;ch7c)C!6KwSpo*t)K`{D<}fg3W~r`D-CeB6ai`lMSxmC5pX)beKpL@1#@%3+*~j>7tGBCb92GmTrf8m%*_RJbHUtPWF;?I$xBx9&M2q?vXYmq<RvS4$x2?bl9#OHy{VuI-dT(ukTQ@eIB{<0=Qj5Vg@6*!M16BL3PGv>hS!YzNv|g4M;w}vKk5A`sQEZ|q!ds9nzX-c)ujDe7fL~@04CRLywj)IcyUmZ@wF~Z#yfqQj2GwmU32lZF3rU|eVU6G8_cf>e5YL#_=WDm|Eh-%#UNDxV{0aUmQOSBo&n9o&xmLuzTT;c_*p(p#Crxb5kDi(=bDG(A?2WXI37|CFuo@99X3to2e~zwAMev-zFQFGAXNY}YZgArr&)NPfM(&x!kUFYme3@8xJ#4pQ9ez=`vf!zKNi*`{INv;X3fFzl!DM498W0-&B5`Mf<QxPV!zU*iTx{HP3+$cYGOYqih__TfPr5rFtBFe7QbfT4Pni|FN<piesWq9@L!yofbaKc0&ekZ0^ShT1pKnNCg3Nh2hY&_n+Pch&A%;v&A%JMntu}^B>}^0lE22LN&ekVP4fTg(Inp&&?NtpkS6(|7)nB_0H!q;m{zmzK_Sh)Uyo|`{YXNy@8cQGzRg)pzBkuu^1aQk$@id;Cf~0|HTixdp~?60j3(da?D2V;dlM-|p}F^<kmla6M>Y2*Qi?)zZz82A&=i{RckyY$e_^dA{1bzk@OO`D!vAAJ6aKOkibARYM%B#Q64T84KwLBL0ZGlgyJj`>z9*-d_v{AEyw97aiTA9qCf=5qCf*0)ns^ULYT})bPf1s3;yt@T6Yuk;tz4ve_pGqy-Ikc<-3Q{DcMnKv-fd(sl!fNqWRS9e(KY!W9MI%HT$ucC3Tg7+E26po>ZB(Bk7V92XbVmLcQ&9bqzYh8&AR8rHS4}1rCE1LTC?so8O^$P)oa$>zd^I^Op|8ae`?jF`>L2G-E-oabl;HDq`M@oN%xwJCf&R0HR<l(ph<V8Nt5nBwQgRfIX9W4Ff`}BA*DHYNm_GmGD%@*&P^sM49&U8B!z*-(7y;QiRfPhPLAnc1V$D9MPOn=|02+%@Gk-#8U2gEy*d4hz~M#|hExFzsTucQQkrr1$Y{pBDXSUxd3BxwLu$r-euHM*qnb72HnnKR9c$Hudr?9Y?!Tlo;qH;qgnLs~6YlfsG~ssUG~qtKK@;v#&6;qVS~TH~<@r|gZ8Az}XueHGDGklH$tb0v`8FA)G&J8fwP?N_GZ<X|QgC@h|59*IT>nyVep3HZ@O<H43MQue7PN-`rC>`||5ET~PXAJ{uIZtI)<9`U6~N#+&!4_h=lQ*>be<nkIL}|XM(6oMt8|`Uyjth^(+cPLvR8GUpIbQ3|GY+8qphGdPG2dlv3J#sg5HqUxN?oO#-UZx8jDv;Yn)cl8fC9aYs@WZji1-NvslmDR(M`AOL5F7C=NX@nWZ@NykwT*(DRa6isPn&;&^BAJB!Z_o*is(ay2-QIDheA4|p9Ov(0R~IPrkjY~%CAiLU~D<~Kfn75KCJj9Tmw{|6!L$=a-s{{_);!OH?uJX1Wi*(Ylk*ts_@IK{)g+U&Pc?s1oU+~wZ4(X)Iz1K80(#Q4t#kvHR5EO2-20p|hdlE@Vy{@dh@@h@X9`CsyPYl;O9I9Xc~x!%WGWBesQYu%bY&#=ZSYpk-yDr=u-uJ+9HV_p5%hOqA8${02y=1pNEeKWiVZ3k_?2Kxr?^WNtj5_=~y$3Mqkmg<){6gm{z(lXL_(1y>i!TY@U42jM0<5QM86vAgqOI;2hyzs#bAH49v3!l2&MZWa`Y;j<11gnb-Phis$|4L(LB_GUU1A;&J_c-@BzmKd99rPabnv#>^r-n`qbz~-{?~dIayCrj3`m5+yQFr@*;2tNw-$xF5@iiq+4dL67xjTmMEt#*P_`2Kox8RF6zIfw{H@<k|i#NXeTdwkZLfE+A#S!fF$c_Z|&%~+>HYxdG1{;@c&S8HK-5FftUE@6^@=5sI+H-54i(ioF6Y3LsD-%o~h#rVGXNof4#J-7L(h#Wcnd+I!%=>$2jhC2HBIni;^IW`7h?uuB2cpC@=lzMfq@ibun9RJ6nZyu93{k`oMGR5R#&;BDh#`uYj+uA)OT$>d@Q4_8MWk08do_N03VSa3cn14A(?5sZn%mKYwMIsU2l@v3`o#}K`uY3$$0lxz?+ES)hSFP7dB}=gp0%aN#Ky#W=g!MLka{3h*VMl$zs+@<o@<Q^^by-H-p^0$*u;(?v7vNpjM&SwV`9Yi&c_j3myaWM({tVD5lb|&L=#Iiu|yL~bZ_H3&&$;r-`T&3*zWT_j1EX)-^4FZW6vd%8EkKQRUP(Y+V~&1$nDGxX~y<9{&gC5(zMc6?D5pU<9CGa2#ro>(;d-{=#|+gvt8p|<8ReHQ}<cov&634jdj;2uTOS1|GQ~q-N?H6t>dOO<Qj4}wT^79YN~1qEPFh4M~EDw^YP@kG9OQlxAO7i*p<6JNsi9uk#*#l-`bEP$4&Woas-xbSws%9j5eMjo4(R`h9|Rc8PD)c9XZHC4ziGgEH@d?FtU{#TNd4)_&$YoN&S++{++(09=k8w)PNmG|Fa%@xBkK=?B=E&Em*ku)ajUM`rq2I-=!_d>PU5@Yh7aoRlc+Snfjs0p~+hte#jk49ZG#ze`kG<RFBkxrUi}5vdgk>wSG5kdv1HKU+ch@qK2Y|iR~3L$Fz)T`R3){rK=<4>RR`3oLoEedB}BZ!=V(pKFsGK*Mg>HS#rJAx;;m(eyv3f<eJz%riEPJyc}FkF0zq}Y&(r-dZwOSWc$!~raSA&MK*Ggja+0S7um=aT>e=ykj0+J+*yx(pSdlEotC?%8GEO$s0sTlcW*OxL({h{*o5Y5r(?nCSF~ftr`u;?joII2cf@wYZmgePKO#9IS(Ljz*OG2Y?`=A#`O>;e>n1jCZMZRWW9IkGvzl7#TI-5h%UTZS4(INizOr>m<C4Y=Gmp)j)iSGP(yS}m&z^qv^na~x%;zKLjrn}!EXwC2=ia7E>&Q8=@x}}}f8X3%N6w<w!#Q%^IekeZIXC3<k#o|lv!|2uU#sW8(r7&6@3J=<&p5rl$au!<bL1o|Imt>+vXYam<Rt3`;~9_5Bqv$PIscUxGq=@a*VUyOu#Q};3G+3ir(u6?a5Q5(8voRS1*hFH9s9Alu?-v9_Mdia=!{M6*sC+rv#=rcbL#sg`z50dM;kiQ9qIW^Bbq1FO{iPYa`&{YxvjbO=1t98>bKNSZdujR(a_P*(E46WtTEPjXKTguvl`E8Y;Aw6-QVhO9W`tCEb4phtQ%(cXzS5-_qrkZyyTAN^OAdh(}X&5FKF4CBX@f~FS#e@^OC!vHP%S(J6q3cBzJ2*FS$qM^OAe*tR8LTzI)x)*M=C+Jf}WtJoC{8a+96hWN$Z~c~kRb<C#~rkelq}CVQ*#%#XE?GM;((EOL{b+*@Bep=P)o9%+_eJW@6nC(Jf!o38@;jGp21SAncg+T^l;w8<H@QaB%kq;P7p3+&Q3Q#{f*-$r#0J;Ocp4EMf`-d!sVaz#iQ<ja^e$j*S2P9!3w^JZL1r(4qjr?k%XK53no{L(s~XS9Z%VGTXQ8hVDc&oeW;(i(jO(i-nXq&4~_q&4RGrFgClN%2(1q<Fk3DV{AY2W`?k_j#px=J=&~4uzz7>T>cy&)nyg>cIy+gAaNJpSs)+erbxeA!&+9acPQ)X=#ef3YublK+0!rM9OD)Ldxe~X(^uvvr<0p_B~E%pMzd$pHoB9K6l5YeZGoH`|NL#FM0-F^vtOtsULjNGx(xs@ZH~XXHeSVld!bI1qo?~U|QOtC?o9<sF!x|grtBjjz|IRNJs&#%18lySWrOCIVqsbyft2FpmS@bf%=4`feu8afxd}J1NBTv19i+)40?tb^b9fR8Dh{g#Gq%0LC+A=F>_>C8X>PN42^JGTpD3ZN*ckImPR-)Cymg*NgAOvEG0A|CMDD>E+uq(N=oSQjFeFSoRrXxCMls!&kgiR3+2Df&_X+c(n75<X`wL%E%ZQ2S}0d1E!2IUVhugxml`4#Jwq&dW=ug15sRK77Cl33_j&)0OABPv(gIIrr3Id;lNPwKpauTjBrPy*nzX>kR%w9&DJi1M(^5pqj1<wTIw>MsPKszqvlP)^r%4f&wn`BNmfaDOChCYv6LpPC6MdGDCb~W;O*FDjny4WsO;pt+O|)f^a-e6(fu11;dWIb68FHX!M%GCckpn$L4)hE;wk)zF#rcgHasD&);`|?S;`}@7#rX>w#rfY&6Xy?X5$9LT6z6}Rk}~=wBV}|+y_8W?gOt%f>!pk?Y?3nC(IRDZ>U1fizqLylee-g4MB3=#xU|vGq_okYl(bQgf;P(Q4?`Pm&q*5<HAowcX^}PxE>|w}47t!V<U-Go3q8}LppM9eo*@@{hFs{GF)dO@<jSj~thjx8y}14QoVfj*W^w!0266kWCUJXNi@1GdtGNBxOmX`a?c(-8Rto9PdMTvaa#BdwG)o~BHAx}e+bo6jZHpArwbP}Lu4tD+vd@x2`q%0mF=?a`Nok~(v^3JCb<#*TW~7l?>!gtm=cJLAG)g1QYLP}dd%85z{8y9{Jws0P3^~y=<V4Sq6Foyt^b9%CGvq|ikP|&aPV@{p=f5(iUL1b3K^#7!SsZ@%G;#Q*W^wqc7IFA{1rD#6E)IXJT^v4qmN@)|+2Ziq>ZO!Y4N^+6CMl)#G$|!Vvy{@GTBMZjm@cK%*e0d)pLQvwO$DWto+YJp_qu*bX{C;|w9<q+X{D_>X{9ao(n=i-(n_&LX{ECYTFKukt#s;iX{8=*(n?!jQ*QJOxzRJ^M$eENJwtBv47t%W<VMe&RZvUhM$eENJwtBv47s<ymRz4)pFeEBe$Iw-Hhj4D!?mw(czwgLwZqm{Z>-)}ysmiN$PFVm)UT~yyLsd0jg!|+UN?8++>M{F{e0~O8!y=S?%H?Pez)<vjT_f*Tz~Gyb2pA(KYo2~LvF*bYkysP^M;!@IM+GXp_DZJIXj3lIx*$p=T`w#6WZzG1bRv8iRsV936v1p2PK5|K?U(UDu~~|jiR2=P^c#~RBaZ$Bo&3iLPatCsm-E%&^Ra`G!DuKjf3jpKB|ZNs2=Wro<T*SrBG35DO40%s#_C!Nooq^g_=Tnp{AJrbZbKCpk+`xXc?3aS_Y+qmO<sPj>=&jmBTtJhxNJ~Y6?w-nnF{drqEQVDKyoV7W9%-)qP&6Dij&23PpyhV*0bC1!aR~LD`^LP&Q~5lnt8YvVvwo)!>J!!4Fk~AF2jFR1JRnTToSKD^wNQ3RQ)+LRF!yP*rFvcRPAX>I$WXx;ix^b%jzxU7^%aS4@B0?I;<v2}%ZSf|5a-pk&Y{C>gW~N(OC$ioqWhgFh+;e^d<qsF-F${YYVT%tT$Gu~1iNEYuYm3w4FY`mms}P*-TI%sljxR2B*jm4$*sWuf3uStvME777lP#q=jL59NXeF{U_i^pZ3P$^{LAazTTjT+kpW7c>Z}g*an|6G!z(X`xz(8&gnPsFvP_>XFjwJ`a_J)<R{WwNP1TEmRg-3zdb|LS><~P+4fLP0yj1q_$9Ys4bKoY71qD+8R?(TPQo!7RnB_#q?*>b0`(G21*63fl@(hpj6NrHx{%8N(HTfQbB8=RL~lz6yi}SqYbqq#f3_F%TPN~T&NV{Q7OcuQf@NT4r&X{h1x=Mp|;Rms4X-XY75PU+Cp=ow$NOtEi_kP8G1>o3&n@(Lh+%xP<*H^6d$T<WSvwOiVxL=;zM;Y{Ru2XnV>09CTI$j37P_Bf~G*3peax$XbO}GngV5lra+mXDNrTkL6wjPRkG7iIZ|Fd3d##rLLO8Jc~B+fL6wjvxE$4m_Cj@`y-;0fFH{%W3)O}8LUp0NP+e#*SQG7q>Oy;c^D=r#>I)@^`a%h!zEFawuO0>UWlVpX&`VNZC_&WMm=>uora#}jj1oaRphVCPC=s*+N(AkI5<xqlM9>Z>5wrtJ1nqzlK|7#C&<?1OMnm05fuTbFznbhjzNvBz;CKOxh$12g1t}s+5acqXfCK5GP$*O=(4C}dcG5KINYiwLHhb@JlBP)q$g(mD-IXDGTu?!l3Wx<o1O!n8xp{cb{p0=g^J!0>yzNQaZ{Hv7xo%Kkpb(e`3W0f`5SRxFfq9^ip69wjfi>&`eSyJ1UtloM7Z?om1qK6sfx$puU@*`Z7!33U1_OP8!9ZVNu;2E8mr!9KL{JzA5flbO1ciYRL17?7P#6di6b3>Bg@F)3VIV|M7@Yq6wg=<^MgVz$5kMYb1ds<90ptNj0C|8BPLM_bd4Lf>9$*BJ2N(h50Y(6IOn<H#lo+T3=7Kt4E~tZet{RjWr~~E>B_#&xa6VTJN(|Hia}T$J!oXsnFt8XX3@io;1B-#ez+#{<uox%|ECvb#i-E$xVqcOL1BHRb;=900s4<WuXbj{C8Us0k#z2msF_0r@4CDwJ8%k;n<Omvr_W)=N<Omvr)1UY*&=M@r5-iXXEYK1x&=M@r5-iXXEYK1x&=M@r5-iXXEYK1xm=5z{I?SI(iVUX1e3%aNVLHr*=`jB|DKeN2^K(d%!E~4p(_#MpU0^X(8H()x=Tz|V=c@nf3>1d7?tUrg3`GXI!v2;~jrDC^1=hEW>fdXzy}@TtX`nBxc#F!g;;pI#rJ=|`QP|zK)L?hBD6zUJs<66ksln=IQU0jI<_2FutzmP6ub|ezYgqRlRbkybT?1M}k%5-5w>_!G-Zou<y^XEG+O|xFwe3kY*0$*itZi(K!Gx_1u0XY6YlACLZJ;fzelw(4{UR%|`n|LPtDi>!YD1BMim<cYtHI93S7T=jk)3UU4lCQ+)mYi?)nH}gtFf|$$jY`rcf)~=4X#AJVPk_UQE%AT;7ZgR_zi2|<|?d#h8nDab`{pZp=!_@iVXCFeQiJu_O&J@_O*j#U$g13uZ=fjUHe9kb!|Wm*0m-j*0qCVU9;)1u8lW`yx7*@5vVw9Yw!qE9JV!^4%-?$0u=|2!%CP@jg_#Y1}oup)mhSUSP3)qpg0s6C<nV*v=X~ox(d6RT#H?8iXOY#>n7}K1}j!IXAM@hXeCy)bQM-LxpqBiIjm~0o3N@G!ZyOD=B&Y{7Oljl29HF|ttT~yP3?6PHZ{X@Rbf4htHgR3y#edtJ2hAj+mu)jo77kjZ|krgE;NDWP-LJP>}hFQ>}k{V*wYRg=8~Ghp4Mo?p4LG2v=}GWv^5&6X=z%lY18yr(+(Q2rkyuoO>4AaO=}=)T1?nN*wSDEst#M)G(EPog9dDAFacGEEv<oUX)(_=g;nuvEmp-lO00@aYOIP^$g222i&gQQ39I4_3s%K`c2FIP4Ag=hEz*D;?VJ%i+CnpSw6jL+Xw6paXa~uT7VX52Mj^1G4boyoi!@+GJ7>g-w$MC;v>jHoW-C^-gJeaEc49@N5Y^kUp}{279X7OcMr>#>33Z1J4JM)Pu%W>u)Ey`aYvWQC*2V}8*2afgtc^)}tc`tItc}NwSR3bCu{Lh7VQu8QKzArI&<Xalxkl_~G!yo-D^~1h6btsVu{P{yG$;16SQqxQ9xwJYst@a#MUVAtt`X}Q&4l&riWTb_#e(&0tPSfK&589a)`j(~$BXrh>TB4A?aZRbb~e|D?Tlu^b_NqsdDzZiA}SBt8B9dwVLO9~s622UR>-wvg}kT33i+Z5E95Z)R><#7SRv<Fu|nqButHXo74oSIl!qb%g<v<UHDfnxuwplBc3?LnY}n2694APHU^k0(VmEUV*v(SNZr16;YBt@7)vVTx)vUpa)vVcp)r_!VHOq5gH8YddEY^wD%t>H1OChUSr|)n(HnZtQY-Y7)Y-TVSwTI1&uwgUHb6_(wlg$h!qxP_wrI5`GCZqPidsruD8L&=<OjsvnWSv}O#yWY%f_1Xmj&<^Z4eR7G_m`yiuuislKzk@MP!pbC4JXg9I(qQ@O4y6%SIc_v{OSOCeib0kuSS#SS9i(ts}a3;eq|!hujcol_tf3!z2T(yI(pE1!d~>=vL5u_0n&Q`(tD#x@7*Q6H=-B4XCl2fzi0n0+&?@6s&5{tKHNV%1F8@Ae}LS7fZYFRa{qV9{lhb$`f&g74EuLQ$a_WQ0z$AOzk%^3eSCgpUMBTnu7|>=6y#EJe$CmEy_|9<XLEKidu_HPyEW_koOxN&%-vasvMt$l8OEHsS!*(9WJYD4N;?!cG5+b|#ij`PDA}CyR|`WzqwqEUdt4rO1^XMegmskpj2^{k$(x$rLTAukr5~dBX>aBYqdlZPpj@Q%QQo9!bJHo(Tw~6y>?zsP*Uw5UiEoZm+HNW$<jqy~vNa{+ie3|S7hDuv;osrC%T41hVeMi@F@4M(%ms|GjNx<(J)5?kO36>7y-)p}R!5sbDWXlJj-`H@`#5WL#=a#L@%`c$wv)=t4H5F0<@3v!r4Nfjg^I#a!sJ4+;2!ToemwUv$H^A6_A!27$(g6~2QmVA`}1A&q`dpIgL(U?1L?m~MbvI;7v(|b@MK<6Y~o7aJu_c@Mj0XRlb2V<OIMeb6z>r~DSRSqD)1Bx;rlt&TpfqT*}`sQE@yqqJe2<x^A<fT|1J7E`7h?V=r&p#RYMt?8=YCXtTizvK8MJ)Y||Z4M^#72pH<3bXQi{t4wZ<D*B2iWn}iDCM8Q`DFYzaGzv8~cm2gh6sjNG!mzh-N)BK^~pFkbmPK%^Xq}`y1GS(+m#E(n(#4om7HF&kknuwYRd3)tw(kW#nCH+b-iq4DYinj|jLW{7Nzg#eY=i;2>MziNLd)W2Na||!z73L_$Y^E&lbl!T}-;`n5G0FBN5lepX{ciu=G}zFks#YusPc&7fRYgmeN+TuPi}n{!7Nr;dEI28o2(}4!3g+=|aTy#To57yOvM}8F5sbe4IJz!xD&-NSBio%iXz7^vn!2;jIBTwXyZSxViz-q0&NNl_%7&F6k&G-^Q<_@*xA^_S9}2%NNEOZ!B=cu-2eKP@k2#w;hneF!dd4M|CSR35IBzL!YEE6s?W9c!&GioFWSh*a)1OjbP;cFk72e~2s*+^8%Ht&CBnc(+i#<i7L?;XXDCjPT7ButayeZsS+zH%Xc97*}EoH_tK4Gk)7w7qNJgb_Mwk8e^?Q&1B-?S_<F4XCC*HnuY{|)bPTGcNVE#;|YZ%BeA6H2BO5rr#6LeX8pR{j~@Ar8TzaL%!JGBa3b7|r2V<umG3+Q-@RQ(sB`Ie`_N?jGi7H4iWw_5W)3s_rP;6z9Sd85J!RYMEAgRw|Zs759lWMY+XugvkYE1&;-%csIG-+_$-rY!mxy#@YP4^!KS{l%muPOY7p@L5=5Ir_naR@|k|J>9J<MW`**NTB0^WKA>uE#R!>S`g&Q5WNS%P$u%)k)Ke(u3x!wtO8&o`Wt`jWBdm{DV_6i&Hu^luN2`;UzLjVV#dx<mUv*?z7aFG;R~v5Wx;3fF)Y>QEJx-Rpszb6z(qt(^@=$WOxVQL`I913lbO|;I{@`xn{LJ3Y{)Ta%o=Lw=eKjj`&CzAuvB8E2f4k?Pon;kSP8dJdZ_{4Yw5ZDzOKUHNU*VIg*yU~Isgi4@=_Ol=W`^H6oWjk8Ckj>xllW=;F}zb8H~S$|$mpLxff`5~y1X-SL_;63!rSG%Z<kqCTP~Vf^qsn2G^MIv6`b&YB;kqG$rU{5Y^ks`w{%BQWN~xhpu)Yvrv+<yAM%*oT+X}f!>p4`dEO}6iM1D!`x4hSrq=a&BVBaYWt+sZ$vn__L{HJZr=qKJ6>jAvd4$|uHKd}ytfuT)*%*nvcy`I0B92HcoGDBU-<eK+FI&Y<W@O|msLXZeR}e|NgXewuUanj2{J<)-jJ6#&vkX^sM$K>PL8>HWYb`sxN1<$H#qjd$(!*u>r4vdAifYB*3O5OJ1$zZUc~QJySgV;*dN=K(jNFt@<A*h*`d505o)XuecD;R$WxV;O>4G6%pQ>q9wP+lw`?Z$v@5gb~%@r%lbEU2_v-F>mABzTwcNaDlh6+XqI(WM{<*aq|8F}uE*py)j>l%r`VPX*xa9wupwgqi7taHo^bCfAfM`)t8r&LcBXKJs8zw1l#w8~)lHJL*)tz>QK9dVcFhL9_K#Q%!h!aK~olD~#FIAci4i;0xRd;aUhc+Xn*2hJysHMSPZE7n~miD83oux69aqODZlR%X^t3g7F>>T~74m0y%jmK>3&ig-m&#Age<!leZY-gj&{>qg$g?3zz^Cv0h47`*MT_GWwkbaGrNw$rvatHl)3M;nLg*BV0F2=!Ri4&~OGn(!XQRhG()<<H9E%TAQMT)a~}Q9P?4k-wimk8^<`q4&!gv}(x`O(Q)Z^w09`_GnzwoX6~~_G`AGwv!gGX|eu{ex>1tW`R1Sq9~^+jo~{pu4-Q8`SM;#P(m$vvpA-RC`b@4=I`JoFtiL=&ahSKiOEer)innGCf@bVc3*Rkbe?f$+SBZ<)>!i(<1*u8JzwwBa@5g^*OkJWtKo^+Rj*YTE25+&l9W<evA1ZxaJzuVjbMuNJ884l2$J4zxD)uSo>sTeSMNFEC0u{Is~s~PYHPK<#o{t&nosGjXbV+5&16-E@>oqHym~6$ET353UOc<>e(}5Fl|rLn12>j6ober%k-8w^SkvJ8pzmYf2i{?x*szjv92cAmU6D46#cf_^eq=sr_}lb??wszTCPF<``FpJ;{F|bw>Z$C%vI)}E(!^q)=ndf;JUOSHv7ZuLTfRiq{Bm$>fa4$Of8eEh33sIHw3FeQ?HFh0*ydW7T3n_I<6DL;+WDF;^<s^<_9talc#oGVS}Q2wJCjmMh_{NIf&rWj%%k+h=`SQMZ<-uDUf)*N<g52iBhGo&xvn^au2If%d%EolOMh#b<z34o<1nKXy0fNFby?Y8F)6&q%T)_y6&0V9y--?HG)Od+KbyUY-nV97+``7F;A?>)fgk<-eM~|~EbuONPjkmP`?+3mw%NCY)tGKsZjLuO3=j16I-O>T{y&Q3+KJ&kX3Acw_^H%Ul3n~#;Z)uL=1OMaI$DCT@uxsu;KRD_e0PaIylmeOE{Z3``@M6IE8Vr#e$dfj4cW5HV=M!VDaN1lJGG~@S5!geAw_feU6@qqk?|zQizr1G1xGo3jFDOUmgH?L4SgM4RR34qZU0T;8Bt8IJsUlb-NW6Z-8URwXQXYZE!XO_yl1^^JZezrhv-|vukqU2KWp2<f4TZqc9k8K4llV{IEkCi^k&~k7B_4P8Ula&IsQEVDQ}c-l4q}XCLwaq_pEl$c7<FC_PdT>ZG-Gg%V{ga$~1gy><I7jOZA&tmg3*=#G{ohWw#`=L~{%F@p^LFl1?<KLYM1D)C&UqfTFJ3FY$E|nf@8X0nc=T?f%xi*xlzybbR34YwfUiT8>zDSn^E_<3C!S&Y_7{$Al+}Dir1ah(!Vh$C~+K(za%C!}WSoeNz47z^uBGI;rnVLg;_bcil6@KiIp#BXz&wS>Z}`$Q&Qrd+n=jV%sm4G*hLaQUAUnMwO~Ih41zC%1e@5Q5L5<_lG54H;-)mJQP<i30@9-7C0Y#SoeiLqJF49NHqH`#2jxcvE8fmEbzYLKIU5Ka5$@+zc{$IN2Wi`LBnm`NR3Q6HoV8pWtT;#1^04yC-pU7XpC%%58e%p2@VVn3hoW)>&^#`*RSypuYc;RuG{ES`8WF~_^Lfqh)&{}x5rcMdg3~7Kkt}q8)rUkdcinJvrZi$pD$Tm_<-(THhb$gpKagxa`Uxj)5bfEiH+wQ=neZD)xl%IdrepBZw6%bae=Svwgtx2Y5b*x(RZ7m5>?()?kC<D*Cpo(tH~BJO*9?RM93eDCgqkTE$lwsv8eraYt5d|+K#reTk>|YzBtliXgRR`@iyAlxm&!O4{T&_e7JFN<G)RchJ?U7^&bag{OSH3-Y8;|caei|HMwTlwwoiv4yUCEV=s5<x@x-|ZOI+AUHtY(dxy7Ww*ApswYPO|%$~cur?gJp724Ue^Zw4t&mVnp=Zn+Zh!)D0-c3>4%$sjDF&kqW3Yz7iS3@uRFNFd-Bjkd({=J#q|91Cu9PPT=`K<j$S9)h!$LP+yt{*$%dJ;OKIv8!Acirjyt7Ca*d;6viTifAIW&1xJi@F1yi`vJuCbj+5Hokp%8-0(u&C+A*j*$NkNx}RN"
# <<PF_BLOB_END>>


def idx3_all():
    """All 3-card hands as an (22100, 3) array in combinadic index order."""
    out = np.empty((N_HANDS3, 3), dtype=I64)
    k = 0
    for c in range(2, 52):
        for b in range(1, c):
            out[k:k + b, 0] = np.arange(b)
            out[k:k + b, 1] = b
            out[k:k + b, 2] = c
            k += b
    return out


def _fallback_pf():
    h = idx3_all()
    r = np.sort(h >> 2, axis=1).astype(float)
    s = h & 3
    score = r[:, 2] * 1.0 + r[:, 1] * 0.7 + r[:, 0] * 0.3
    pair = (r[:, 0] == r[:, 1]) | (r[:, 1] == r[:, 2])
    score += pair * (6.0 + r[:, 1] * 0.6)
    suited = (s[:, 0] == s[:, 1]) | (s[:, 1] == s[:, 2]) | (s[:, 0] == s[:, 2])
    score += suited * 1.5
    eq = 0.3 + 0.4 * (score - score.min()) / (score.max() - score.min())
    i = (np.arange(NB) + 0.5) / NB
    e = 0.5 + 0.30 * (i[:, None] - i[None, :])
    return eq, e


def _load_pf():
    if _PF_BLOB:
        raw = zlib.decompress(base64.b85decode(_PF_BLOB))
        eq = np.frombuffer(raw[:N_HANDS3 * 2], dtype="<u2").astype(float) / 65535.0
        e = np.frombuffer(raw[N_HANDS3 * 2:], dtype="<u2").astype(float).reshape(NB, NB) / 65535.0
    else:
        eq, e = _fallback_pf()
    srt = np.sort(eq)
    pct = (np.searchsorted(srt, eq, "left") + np.searchsorted(srt, eq, "right")) * (0.5 / eq.size)
    bkt = np.minimum(NB - 1, (pct * NB).astype(I64))
    return eq, pct, bkt, e


PF_EQ, PF_PCT, PF_BKT, PF_E = _load_pf()
BIN_C = (np.arange(NB) + 0.5) / NB
KEEPS = ((0, 1), (0, 2), (1, 2))          # keep index k discards column 2 - k
KEEP_ARR = np.array(KEEPS, dtype=I64)
_TRIU = {}


def pairs_of(cards):
    """All unordered pairs of the given card ids as an (M, 2) array."""
    n = len(cards)
    iu = _TRIU.get(n)
    if iu is None:
        iu = np.triu_indices(n, 1)
        _TRIU[n] = iu
    u = np.asarray(cards, dtype=I64)
    return np.column_stack((u[iu[0]], u[iu[1]]))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def wpct(strength, w):
    """Weighted percentile (0 weakest .. 1 strongest) of each entry inside the range."""
    order = np.argsort(strength, kind="stable")
    ws = w[order]
    cw = np.cumsum(ws)
    tot = cw[-1]
    q = np.empty(strength.shape[0])
    if tot <= 0:
        q[order] = (np.arange(q.size) + 0.5) / q.size
        return q
    q[order] = (cw - 0.5 * ws) / tot
    return q


def flop_ref(flop):
    """Sorted strengths of every 2-card holding on this flop (reference distribution)."""
    fs = set(flop)
    rest = [c for c in range(52) if c not in fs]
    h = pairs_of(rest)
    full = np.column_stack((h, np.broadcast_to(np.array(flop, dtype=I64), (h.shape[0], 3))))
    return np.sort(evalhands(full))


def flop_value(cards5, ref):
    """Heuristic flop value in [0,1]: made-hand percentile lifted by draws."""
    v, m1, sm = evalhands(cards5, aux=True)
    pct = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) * (0.5 / ref.size)
    fd = np.zeros(v.shape[0], dtype=bool)
    for j in range(4):
        fd |= POP[(sm >> (16 * j)) & 0x1FFF] == 4
    so = SOUT[m1]
    d = 0.36 * fd + np.where(so >= 2, 0.30, np.where(so == 1, 0.12, 0.0))
    d = np.minimum(d, 0.55)
    return pct + (1.0 - pct) * d


def lik(spec, q):
    """Likelihood of an observed opponent action for range percentile q."""
    kind = spec[0]
    if kind == "mix":               # (1 - h) * spec_a + h * spec_b
        _, h, sa, sb = spec
        return (1.0 - h) * lik(sa, q) + h * lik(sb, q)
    if kind == "band":              # an honest sizer: this bet size comes from one slice of the range
        _, lo, hi, tau = spec
        return np.maximum(sig((q - lo) / tau) * (1.0 - sig((q - hi) / tau)), 0.03)
    if kind == "agg":
        if len(spec) == 5:          # polarised: value from the top, bluffs from the bottom
            _, av, bl, tau, bf = spec
            return np.maximum(np.maximum(sig((q - (1.0 - av)) / tau), sig((bf - q) / tau)), 0.02)
        _, av, bl, tau = spec
        return np.maximum(sig((q - (1.0 - av)) / tau), bl)
    if kind == "call":
        _, fold, ar, tau = spec
        return np.maximum(sig((q - fold) / tau) * (1.0 - P["slow"] * sig((q - (1.0 - ar)) / tau)), 0.02)
    _, av, dep, tau = spec
    return 1.0 - dep * sig((q - (1.0 - av)) / tau)


# ------------------------------------------------------------ history replay

class Replay:
    """Re-plays the action history to recover pot, bets and the context of each action."""

    def __init__(self, i_am_button):
        self.street = "preflop"
        self.bet = [1, 2] if i_am_button else [2, 1]     # [me, opp]
        self.pot = 3
        self.nr = 0
        self.agg = [False, False]
        self.calls = [0, 0]                              # post-flop calls made so far

    def step(self, e):
        street = e.get("street", self.street)
        if street != self.street:
            self.street = street
            self.bet = [0, 0]
            self.nr = 0
            self.agg = [False, False]
        i = 0 if e.get("actor") == "me" else 1
        o = 1 - i
        action = e.get("action")
        c = self.bet[o] - self.bet[i]
        pot = self.pot
        if street == "preflop":
            if self.nr == 0:
                ctx = "p_open" if c > 0 else "p_vslimp"
            elif self.nr == 1:
                ctx = "p_vsopen"
            elif self.nr == 2:
                ctx = "p_vs3bet"
            else:
                ctx = "p_vs4bet"
        else:
            ctx = "open" if c == 0 else ("facing_raise" if self.agg[i] else "facing")
        info = {"street": street, "who": i, "action": action, "ctx": ctx, "cls": 0,
                "P": pot, "c": c, "sf": (c / max(1.0, pot - c)) if c > 0 else 0.0, "amt": 0,
                "k": min(2, self.calls[i])}
        if action == "call":
            if street != "preflop":
                self.calls[i] += 1
            self.pot += c
            self.bet[i] = self.bet[o]
        elif action == "raise":
            amt = int(e.get("amount", self.bet[o]))
            prev = self.bet[o]
            if street == "preflop":
                cls = 2 if amt >= 80 else (1 if amt > max(8.0, 4.5 * prev) else 0)
            else:
                s = (amt - prev) / max(1.0, pot + c)
                cls = 0 if s < 0.45 else (1 if s < 1.05 else (2 if s < 2.2 else 3))
                info["s"] = s
            info["cls"] = cls
            info["amt"] = amt
            self.pot += amt - self.bet[i]
            self.bet[i] = amt
            self.nr += 1
            self.agg[i] = True
        return info


# ------------------------------------------------------------ opponent model

PRIORS = {      # [fold, passive (call/check), raise class 0, 1, 2, 3]
    "p_open":       [0.08, 0.14, 0.68, 0.07, 0.03, 0.00],
    "p_vslimp":     [0.00, 0.62, 0.30, 0.05, 0.03, 0.00],
    "p_vsopen":     [0.27, 0.58, 0.11, 0.02, 0.02, 0.00],
    "p_vs3bet":     [0.42, 0.44, 0.08, 0.02, 0.04, 0.00],
    "p_vs4bet":     [0.50, 0.46, 0.00, 0.00, 0.04, 0.00],
    "open":         [0.00, 0.55, 0.14, 0.22, 0.07, 0.02],
    "facing":       [0.40, 0.48, 0.05, 0.04, 0.02, 0.01],
    "facing_raise": [0.45, 0.43, 0.04, 0.04, 0.02, 0.02],
}
PRIOR_N = {"p_open": 8.0, "p_vslimp": 6.0, "p_vsopen": 8.0, "p_vs3bet": 5.0, "p_vs4bet": 4.0,
           "open": 7.0, "facing": 7.0, "facing_raise": 4.0}

# Strategy constants (tuned with tools/ab.py).
P = {
    "fold_k": 0.85,        # default fold frequency to a bet of s pots = fold_k * s / (1 + s)
    "fold_m1": 0.8,        # ... scaled down when the range has already called once
    "fold_m2": 0.6,        # ... or twice in this hand
    "bet_margin": 0.03,    # a bet must beat checking by this fraction of the pot
    "oop_pen": 0.05,       # value lost by checking out of position (fraction of pot)
    "bluff_ip": 0.32,      # base river bluff frequency in / out of position
    "bluff_oop": 0.24,
    "semi_prob": 0.45,     # frequency of close-EV semi-bluffs on flop and turn
    "semi_slack": 0.08,
    "r_ip": 0.97,          # share of equity realised when calling with streets to come
    "r_oop": 0.92,
    "vraise_eqc": 0.56,    # value raise needs this equity against the continuing range
    "vraise_margin": 0.03,
    "big_call_margin": 0.02,   # extra edge demanded for calls above 40 chips (doubled above 100)
    "pf_open": 0.12,       # button opens hands above this percentile
    "pf_iso": 0.60,        # big blind raises limpers above this percentile
    "pf_3bet_eq": 0.54,    # re-raise for value: equity vs the top half of their range
    "pf_3bet_p": 0.70,
    "pf_call_margin": 0.02,
    "pf_allin_margin": 0.01,   # extra equity demanded to call an all-in before the flop (0.05 folded +EV calls)
    "risk_sd": 30.0,           # chips of standard deviation per hand, for lead-dependent risk control
    "size_read": 1,            # 1 = learn whether the opponent's bet size tells its strength (see OppModel.honesty)
    "honest_d0": 0.08,         # ... shown-strength gap (big bets minus small bets) where the size read starts
    "honest_d1": 0.22,         # ... and the gap where it is fully trusted
    "pf_r_ip": 0.97,
    "pf_r_oop": 0.92,
    "pf_r_slope": 0.25,    # weak hands realise less of their equity
    "slow": 0.7,           # how strongly a call / check discounts the top of a range
    "tau": 0.06,
    "beta0": 0.25,         # prior share of bluffs in a betting range
    "big_s": 0.8,          # a turn / river bet of at least this many pots is "big"
    "gto_bluff": 1,        # 1 = big bets start from the game-theory bluff share s / (1 + 2s)
    "polar": 1,            # 1 = river bets are modelled as value (top of range) plus bluffs (bottom)
    "polar_turn": 1,       # ... and turn bets too
    "turn_thin": 0.5,      # weak hands keep a big (0.8+ pot) turn bet only this often; else smaller or check
    "turn_bluff_eq": 0.40, # "weak" for the rule above: equity below this
    "bluff_pct": 0.40,     # a shown-down bet is a bluff if weaker than this share of all holdings
    "value_max": 0.45,     # most often an honest range bets the turn / river when checked to
    "vraise_max": 0.15,    # ... or raises a bet there
    "max_overbet": 1.6,    # all-in is considered when it is at most this many pots
    "solve_pot": 20,       # river pots at least this big are played with the CFR solver
    "solve_cap": 380,      # most hands kept in each range for the solver
    "purify": 0.10,        # solver actions below this frequency are dropped
    "solve_mode": 0,       # 0 = every river spot, 1 = only when checked to, 2 = only facing a bet
    "cap_exit": -60,       # capped mode switches to fold-only after losing this many chips
    "n_sim": 6000,           # flop Monte Carlo deals (2500 was enough; more only costs time, ~38 s of 150 s)
}


def fold_base(s):
    return P["fold_k"] * s / (1.0 + s)


class OppModel:
    def __init__(self):
        self.cnt = {k: np.zeros(6) for k in PRIORS}
        self.n = {k: 0.0 for k in PRIORS}
        self.pri = {k: np.array(v, dtype=float) for k, v in PRIORS.items()}
        self.fold_obs = np.zeros(3)         # indexed by bets already called this hand
        self.fold_exp = np.zeros(3)
        self.bluff_n = 0.0
        self.bluff_k = 0.0
        self.bigb_n = 0.0               # showdowns after a big turn / river bet ...
        self.bigb_k = 0.0               # ... and how many of those were bluffs
        # chances to bet ("open") or raise ("facing") on the turn / river, and how many were taken
        self.agg_n = {(s, c): 0.0 for s in ("turn", "river") for c in ("open", "facing")}
        self.agg_k = {k: 0.0 for k in self.agg_n}
        self.pi = np.array([0.60, 0.20, 0.20])     # discard habit: best keep / lowest card / random
        # strength (percentile among all holdings on that street's board) of hands shown down after
        # this opponent's post-flop bets, by size: [small (<0.45 pot), medium, big (>1.05 pot)]
        self.sz_n = np.zeros(3)
        self.sz_sum = np.zeros(3)
        self.sz_sq = np.zeros(3)

    def freq(self, ctx):
        n0 = PRIOR_N[ctx]
        return (self.cnt[ctx] + n0 * self.pri[ctx]) / (self.n[ctx] + n0)

    def obs(self, ctx, action, cls):
        if ctx not in self.cnt:
            return
        self.n[ctx] += 1.0
        if action == "fold":
            self.cnt[ctx][0] += 1.0
        elif action in ("call", "check"):
            self.cnt[ctx][1] += 1.0
        elif action == "raise":
            self.cnt[ctx][2 + min(3, max(0, cls))] += 1.0

    def fold_mult(self, k=0):
        m0 = (1.0, P["fold_m1"], P["fold_m2"])[k]
        return float((self.fold_obs[k] + 2.5 * m0) / (self.fold_exp[k] + 2.5))

    def fold_frac(self, s, k=0):
        return float(min(0.92, max(0.02, fold_base(s) * self.fold_mult(k))))

    def honesty(self):
        """0 .. 1: how strongly this opponent's bet size reveals its hand (big bets = strong hands,
        small bets = medium hands), from showdowns.  0 until there is clear evidence."""
        if not P["size_read"]:
            return 0.0
        n_hi = self.sz_n[1] + self.sz_n[2]          # medium and big bets pooled
        m = min(self.sz_n[0], n_hi)
        if m < 4:
            return 0.0
        n_lo = self.sz_n[0]
        mu_hi = (self.sz_sum[1] + self.sz_sum[2]) / n_hi
        mu_lo = self.sz_sum[0] / n_lo
        var_hi = max((self.sz_sq[1] + self.sz_sq[2]) / n_hi - mu_hi * mu_hi, 0.01)
        var_lo = max(self.sz_sq[0] / n_lo - mu_lo * mu_lo, 0.01)
        d = mu_hi - mu_lo
        z = d / (var_hi / n_hi + var_lo / n_lo) ** 0.5       # is the gap more than chance?
        h = (d - P["honest_d0"]) / (P["honest_d1"] - P["honest_d0"])
        return float(min(1.0, max(0.0, h)) * min(1.0, max(0.0, (z - 2.0) / 1.5)))

    def beta_big(self, s):
        """Bluff share of a big bet of s pots: the game-theory share s / (1 + 2s) as the prior,
        moved by showdowns seen after this opponent's big bets.  A balanced opponent bluffs this
        often, so calling against it is break-even; only evidence of honesty makes us fold more."""
        prior = min(0.45, s / (1.0 + 2.0 * s))
        return float((self.bigb_k + prior * 6.0) / (self.bigb_n + 6.0))

    def beta(self, street=None):
        """Estimated share of bluffs in the opponent's betting range.

        Two estimates, the larger wins: hands shown down after a turn/river bet, and,
        for the turn and river, the betting frequency itself.  The second needs no
        showdown, so it still works when we keep folding to the bets."""
        b = (self.bluff_k + P["beta0"] * 5.0) / (self.bluff_n + 5.0)
        if street in ("turn", "river"):
            for ctx, vmax in (("open", P["value_max"]), ("facing", P["vraise_max"])):
                key = (street, ctx)
                f = (self.agg_k[key] + 0.8 * vmax * 6.0) / (self.agg_n[key] + 6.0)
                b = max(b, (f - vmax) / f)
        return float(min(0.75, max(0.08, b)))


# --------------------------------------------------------------- river solver

class _Node:
    __slots__ = ("kind", "p", "kids", "acts", "bets", "loser", "R", "S", "s")

    def __init__(self, kind, p, bets):
        self.kind = kind          # "d" decision, "s" showdown, "f" fold
        self.p = p                # player to act (0 = us, 1 = opponent) or the player who folded
        self.kids = []
        self.acts = []            # ("fold" | "call" | "check" | "raise", raise-to amount)
        self.bets = bets          # street bets [us, opponent] on reaching this node


class RiverSolver:
    """Range-vs-range river subgame solved with discounted CFR (alpha 1.5, beta 0, gamma 2).

    Hands are rows of (n, 2) card arrays with weights and 7-card strengths.  Showdown and
    fold values for every terminal are computed with one batched matrix product per
    iteration (blocked hand pairs are masked out), so an iteration costs a few
    milliseconds for ranges of a few hundred hands.
    """

    def __init__(self, hands, weights, strengths, base, bets, behind, acted, raises_left, last_raise):
        bits = [np.left_shift(_ONE, h[:, 0]) | np.left_shift(_ONE, h[:, 1]) for h in hands]
        self.K = ((bits[0][:, None] & bits[1][None, :]) == 0).astype(float)
        self.M = np.sign(strengths[0][:, None] - strengths[1][None, :]) * self.K
        self.w = [np.asarray(weights[0], float), np.asarray(weights[1], float)]
        self.base = float(base)
        self.bets0 = list(bets)
        self.beh0 = list(behind)
        self.nodes = []
        self._build(list(bets), 0, list(acted), raises_left, last_raise, 0)

    def _new(self, kind, p, bets):
        nd = _Node(kind, p, bets)
        self.nodes.append(nd)
        return len(self.nodes) - 1, nd

    def _build(self, bets, p, acted, nr, last, depth):
        o = 1 - p
        idx, nd = self._new("d", p, list(bets))
        c = bets[o] - bets[p]
        beh = [self.beh0[i] - (bets[i] - self.bets0[i]) for i in (0, 1)]
        maxto = min(bets[p] + beh[p], bets[o] + beh[o])
        if c > 0:
            k, _ = self._new("f", p, list(bets))
            nd.acts.append(("fold", None))
            nd.kids.append(k)
            nb = list(bets)
            nb[p] = bets[o]
            k, _ = self._new("s", p, nb)
            nd.acts.append(("call", None))
            nd.kids.append(k)
        else:
            nd.acts.append(("check", None))
            if acted[o]:
                k, _ = self._new("s", p, list(bets))
            else:
                na = list(acted)
                na[p] = True
                k = self._build(list(bets), o, na, nr, last, depth)
            nd.kids.append(k)
        if nr > 0 and maxto > bets[o]:
            pot = 2.0 * self.base + bets[0] + bets[1]
            mn = bets[o] + max(last, 2)
            want = []
            if depth < 2:
                want = [0.5 * pot, 1.0 * pot] if c == 0 else [bets[o] + (pot + c)]
            sizes = set()
            for to in want:
                to = int(round(max(mn, min(maxto, to))))
                if to - bets[p] >= 0.85 * (maxto - bets[p]):
                    to = maxto
                sizes.add(to)
            sizes.add(int(maxto))
            for to in sorted(sizes):
                nb = list(bets)
                nb[p] = to
                na = list(acted)
                na[p] = True
                k = self._build(nb, o, na, nr - 1, to - bets[o], depth + 1)
                nd.acts.append(("raise", to))
                nd.kids.append(k)
        return idx

    def solve(self, iters, deadline):
        nodes = self.nodes
        n = (self.w[0].size, self.w[1].size)
        dec = [i for i, nd in enumerate(nodes) if nd.kind == "d"]
        show = [i for i, nd in enumerate(nodes) if nd.kind == "s"]
        fold = [i for i, nd in enumerate(nodes) if nd.kind == "f"]
        for i in dec:
            nd = nodes[i]
            a = len(nd.kids)
            nd.R = np.zeros((a, n[nd.p]))
            nd.S = np.zeros((a, n[nd.p]))
        c_show = np.array([self.base + nodes[i].bets[0] for i in show])
        c_fold = np.array([self.base + nodes[i].bets[nodes[i].p] for i in fold])
        s_fold = np.array([1.0 if nodes[i].p == 1 else -1.0 for i in fold])   # sign of our value
        M, K, MT, KT = self.M, self.K, self.M.T, self.K.T
        done = 0
        for t in range(1, iters + 1):
            if done >= 8 and time.monotonic() > deadline:
                break
            reach = [None] * len(nodes)
            reach[0] = self.w
            for i in dec:
                nd = nodes[i]
                pos = np.maximum(nd.R, 0.0)
                tot = pos.sum(0)
                nd.s = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / pos.shape[0])
                r = reach[i]
                for a, k in enumerate(nd.kids):
                    rk = [r[0], r[1]]
                    rk[nd.p] = r[nd.p] * nd.s[a]
                    reach[k] = rk
            vals = [None] * len(nodes)
            if show:
                r0 = np.stack([reach[i][0] for i in show], 1)
                r1 = np.stack([reach[i][1] for i in show], 1)
                v0 = (M @ r1) * c_show
                v1 = -(MT @ r0) * c_show
                for j, i in enumerate(show):
                    vals[i] = (v0[:, j], v1[:, j])
            if fold:
                r0 = np.stack([reach[i][0] for i in fold], 1)
                r1 = np.stack([reach[i][1] for i in fold], 1)
                v0 = (K @ r1) * (c_fold * s_fold)
                v1 = -(KT @ r0) * (c_fold * s_fold)
                for j, i in enumerate(fold):
                    vals[i] = (v0[:, j], v1[:, j])
            pw = t ** 1.5
            dpos = pw / (pw + 1.0)
            gw = (t / (t + 1.0)) ** 2
            for i in reversed(dec):
                nd = nodes[i]
                p = nd.p
                vp = np.stack([vals[k][p] for k in nd.kids])
                vo = vals[nd.kids[0]][1 - p].copy()
                for k in nd.kids[1:]:
                    vo += vals[k][1 - p]
                vn = (nd.s * vp).sum(0)
                nd.R += vp - vn
                nd.R *= np.where(nd.R > 0, dpos, 0.5)
                nd.S = nd.S * gw + reach[i][p] * nd.s
                vals[i] = (vn, vo) if p == 0 else (vo, vn)
            done = t
        return done

    def root_strategy(self, hand_index):
        nd = self.nodes[0]
        col = nd.S[:, hand_index]
        tot = col.sum()
        if tot <= 0:
            return nd.acts, np.full(len(nd.acts), 1.0 / len(nd.acts))
        return nd.acts, col / tot


# ----------------------------------------------------------------------- bot

class Bot:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.opp = OppModel()
        self.me = OppModel()            # our own tendencies, as the opponent could measure them
        self.solves = 0
        self.total_hands = 400
        self.hand_id = 0
        self.net = 0
        self.locked = False
        self.risk = 0.0
        self.free_n = 0                 # hands played without the lock, and their net sums
        self.free_s1 = 0.0
        self.free_s2 = 0.0
        self.cap = None                 # chip limit for this hand while the win is locked in
        self.capped_net = 0.0           # result of hands played under such a limit
        self.mu_est = 0.0
        self.start_stack = 200
        self.time_left = 60000.0
        self.n_sim = 2000
        self.river_step = 1
        self.errors = 0
        self._reset_hand()

    # ----------------------------------------------------------- hand state
    def _reset_hand(self):
        self.button = True
        self.my3 = []
        self.my_hole = []
        self.my_disc = None
        self.opp_disc = None
        self.flop = None
        self.cur_board = []
        self.wp = np.ones(NB)
        self.rp = None
        self.hist_i = 0
        self.flop_specs = []
        self.sim = None
        self.ref = None
        self.H = None
        self.W = None
        self.HV = None
        self.t_ready = False
        self.r_ready = False
        self.t_winp = self.t_str = None
        self.r_winp = self.r_str = None
        self.my_pct = 0.5
        self.my_bkt = NB // 2

    def on_init(self, msg):
        self.total_hands = int(msg.get("total_hands", 400))
        self.time_left = float(msg.get("time_bank_ms", 60000))

    def on_new_hand(self, msg):
        self._reset_hand()
        self.hand_id = int(msg.get("hand_id", self.hand_id + 1))
        self.button = msg.get("seat", "button") == "button"
        self.my3 = [cid(c) for c in msg.get("hole_cards", [])]
        s = sorted(self.my3)
        k = s[0] + int(C2T[s[1]]) + int(C3T[s[2]])
        self.my_pct = float(PF_PCT[k])
        self.my_bkt = int(PF_BKT[k])
        self.rp = Replay(self.button)
        # match lock: lead larger than every blind we can still be forced to post
        # (worst case per hand: 1 chip on the button, 2 in the big blind).  While the
        # lead also covers a full 200-chip loss in this hand we keep playing normally.
        after = max(0, self.total_hands - self.hand_id)
        nxt = (after + 1) // 2                      # hands in the opposite seat
        same = after // 2
        cost_rest = nxt * (2 if self.button else 1) + same * (1 if self.button else 2)
        budget = self.net - cost_rest - 1
        left = max(1, self.total_hands - self.hand_id + 1)
        blind = 1 if self.button else 2
        self.start_stack = int(msg.get("stack", 200))
        self.locked = False
        self.cap = None
        if LOCK and blind <= budget < 200 and self._should_lock(left, self.net - cost_rest - blind):
            # The win is now certain as long as this hand costs at most `budget` chips.
            # With evidence of an edge we keep playing inside that limit (the surplus is
            # only tie-break chips); otherwise we simply check or fold.
            if self.mu_est >= 2.0 and budget >= 6 and self.capped_net > P["cap_exit"]:
                self.cap = budget
            else:
                self.locked = True
        # risk attitude: ahead -> avoid coin flips for stacks, far behind -> welcome them
        self.risk = float(np.clip(self.net / (P["risk_sd"] * left ** 0.5), -3.0, 3.0))

    def _should_lock(self, left, guaranteed):
        """Lock the match (check / fold only) or keep playing?

        Locking makes the win certain but freezes the margin, which costs tie-break
        chips (capped chips per hand).  We estimate the chance of still winning if we
        play on, from the per-hand results of this match (normal model, edge shrunk
        towards zero), and lock when the win probability given up is worth more than
        the tie-break chips we expect to add.
        """
        h = self.free_n
        self.mu_est = 0.0
        if h < 8:
            return True
        # chips per match point: match points always rank first, and the capped mode
        # below collects tie-break chips without risk, so playing on must be near-certain
        lam = 60000.0 if self.total_hands <= 600 else 150000.0
        cap = float(self.total_hands)                             # C = 400 / 1000
        mean = self.free_s1 / h
        sd = 1.2 * max(25.0, (max(0.0, self.free_s2 / h - mean * mean)) ** 0.5)
        prec = h / (sd * sd) + 1.0 / 144.0                        # prior on the edge: sd 12 chips/hand
        mu = (mean * h / (sd * sd)) / prec
        self.mu_est = mu
        final = self.net + mu * left
        var = sd * sd * left + left * left / prec
        p_win = 0.5 * (1.0 + math.erf(final / (2.0 * var) ** 0.5))
        tb_gain = max(0.0, min(cap, final) - guaranteed)
        return (1.0 - p_win) * lam > tb_gain

    def _budget(self, req):
        tl = req.get("time_left_ms")
        if tl is not None:
            self.time_left = float(tl)
        left = max(1, self.total_hands - self.hand_id + 1)
        per = self.time_left / left
        if per > 60:
            self.n_sim, self.river_step = int(P["n_sim"]), 1
        elif per > 30:
            self.n_sim, self.river_step = 1500, 1
        elif per > 14:
            self.n_sim, self.river_step = 700, 2
        else:
            self.n_sim, self.river_step = 250, 4
        return per

    # --------------------------------------------------------- range updates
    def sync(self, history):
        while self.hist_i < len(history):
            e = history[self.hist_i]
            self.hist_i += 1
            info = self.rp.step(e)
            if info["who"] == 1:
                spec = self.make_spec(info)
                if spec is not None:
                    self.apply_spec(info["street"], spec)

    def make_spec(self, info, st=None):
        if st is None:
            st = self.opp
        f = st.freq(info["ctx"])
        a = info["action"]
        pre = info["street"] == "preflop"
        if a == "raise":
            big = float(f[2 + info["cls"]:].sum())
            big = min(max(big, 0.005), 0.98)
            if pre:
                return ("agg", big, min(0.06, 0.3 * big / (1.0 - big + 1e-6)), 0.05)
            be = st.beta(info["street"])
            s_bet = info.get("s", 0.0)
            if P["gto_bluff"] and s_bet >= P["big_s"]:
                be = max(be, st.beta_big(s_bet))
            av = min(max(big * (1.0 - be), 0.01), 0.97)
            bl = min(1.0, max(0.02, big * be / (1.0 - av)))
            if P["polar"] and (info["street"] == "river" or (P["polar_turn"] and info["street"] == "turn")):
                spec = ("agg", av, bl, P["tau"], min(0.6, big * be))
            else:
                spec = ("agg", av, bl, P["tau"])
            h = st.honesty()
            if h > 0.0:
                cls = info["cls"]
                above = float(f[3 + cls:].sum()) if cls < 3 else 0.0
                above = min(above, big - 0.01)
                spec = ("mix", h, spec, ("band", 1.0 - big, 1.0 - above if above > 0.005 else 2.0, P["tau"]))
            return spec
        if a == "call":
            ar = float(f[2:].sum())
            if pre:
                return ("call", float(f[0]), ar, 0.05)
            fold = st.fold_frac(info["sf"], info["k"]) if info["ctx"] == "facing" else float(f[0])
            return ("call", fold, ar, P["tau"])
        if a == "check":
            ar = float(f[2:].sum())
            if pre:
                return ("check", ar, 0.8, 0.05)
            return ("check", ar * (1.0 - st.beta(info["street"])), P["slow"], P["tau"])
        return None

    def apply_spec(self, street, spec):
        if street == "preflop":
            self.wp = self.wp * lik(spec, wpct(BIN_C, self.wp))
            self.wp = self.wp / max(1e-12, self.wp.max())
        elif street == "flop":
            if self.H is None:
                self.flop_specs.append(spec)
                if self.sim is not None:
                    self._sim_apply(spec)
            else:
                self._w_apply(self.HV, spec)
        elif street == "turn":
            if self.H is not None and len(self.cur_board) >= 4:
                self.ensure_turn()
                self._w_apply(self.t_str, spec)
        elif street == "river":
            if self.H is not None and len(self.cur_board) >= 5:
                self.ensure_river()
                self._w_apply(self.r_str, spec)

    def _w_apply(self, strength, spec):
        w = self.W * lik(spec, wpct(strength, self.W))
        m = w.max()
        self.W = w / m if m > 0 else np.ones_like(w)

    def _sim_apply(self, spec):
        s = self.sim
        w = s["w"] * lik(spec, wpct(s["V"], s["w"]))
        m = w.max()
        s["w"] = w / m if m > 0 else np.ones_like(w)

    # ------------------------------------------------------- flop simulation
    def ensure_sim(self, flop):
        if self.sim is not None:
            return
        self.flop = list(flop)
        n = self.n_sim
        seen = set(self.my3) | set(flop)
        u = np.array([c for c in range(52) if c not in seen], dtype=I64)
        idx = np.argsort(self.rng.random((n, u.size)), axis=1)[:, :5]
        cs = u[idx]
        opp3 = cs[:, :3]
        run = cs[:, 3:]
        s3 = np.sort(opp3, axis=1)
        wpre = self.wp[PF_BKT[s3[:, 0] + C2T[s3[:, 1]] + C3T[s3[:, 2]]]]
        fl = np.broadcast_to(np.array(flop, dtype=I64), (n, 3))
        if self.ref is None:
            self.ref = flop_ref(flop)
        vs = np.stack([flop_value(np.column_stack((opp3[:, i], opp3[:, j], fl)), self.ref)
                       for i, j in KEEPS])
        kbest = vs.argmax(0)
        klow = 2 - opp3.argmin(1)
        krnd = self.rng.integers(0, 3, n)
        hyp = self.rng.choice(3, size=n, p=self.opp.pi / self.opp.pi.sum())
        ksel = np.where(hyp == 0, kbest, np.where(hyp == 1, klow, krnd))
        opp2 = np.take_along_axis(opp3, KEEP_ARR[ksel], axis=1)
        board = np.concatenate((fl, run), axis=1)
        ov = evalhands(np.concatenate((opp2, board), axis=1))
        outs = []
        for i, j in KEEPS:
            mine = np.concatenate(
                (np.broadcast_to(np.array([self.my3[i], self.my3[j]], dtype=I64), (n, 2)), board), axis=1)
            mv = evalhands(mine)
            outs.append((mv > ov) + 0.5 * (mv == ov))
        self.sim = {"w": np.array(wpre, dtype=float), "out": np.stack(outs), "V": vs.max(0)}
        for spec in self.flop_specs:
            self._sim_apply(spec)

    def keep_equities(self):
        s = self.sim
        tot = s["w"].sum()
        if tot <= 0:
            return s["out"].mean(axis=1)
        return (s["out"] * s["w"]).sum(axis=1) / tot

    # ------------------------------------------------ post-discard range init
    def init_range(self):
        flop = self.flop
        d = self.opp_disc
        dead = set(self.my3) | set(flop) | {d}
        u = [c for c in range(52) if c not in dead]
        h = pairs_of(u)
        m = h.shape[0]
        dcol = np.full(m, d, dtype=I64)
        tri = np.sort(np.column_stack((h, dcol)), axis=1)
        wpre = self.wp[PF_BKT[tri[:, 0] + C2T[tri[:, 1]] + C3T[tri[:, 2]]]]
        fl = np.broadcast_to(np.array(flop, dtype=I64), (m, 3))
        if self.ref is None:
            self.ref = flop_ref(flop)
        vh = flop_value(np.column_stack((h, fl)), self.ref)
        va = flop_value(np.column_stack((h[:, 0], dcol, fl)), self.ref)
        vb = flop_value(np.column_stack((h[:, 1], dcol, fl)), self.ref)
        eps = 0.10
        l1 = eps / 2 + (1.0 - 1.5 * eps) * sig((vh - np.maximum(va, vb)) / 0.03)
        dr = d >> 2
        l2 = np.where(((h[:, 0] >> 2) >= dr) & ((h[:, 1] >> 2) >= dr), 1.0 - eps, eps / 2)
        pi = self.opp.pi / self.opp.pi.sum()
        w = wpre * (pi[0] * l1 + pi[1] * l2 + pi[2] / 3.0)
        self.H, self.W, self.HV = h, np.array(w, dtype=float), vh
        for spec in self.flop_specs:
            self._w_apply(self.HV, spec)

    def _filter(self, card):
        keep = (self.H[:, 0] != card) & (self.H[:, 1] != card)
        self.H, self.W, self.HV = self.H[keep], self.W[keep], self.HV[keep]
        return keep

    def ensure_turn(self):
        if self.t_ready:
            return
        b4 = self.cur_board[:4]
        self._filter(b4[3])
        h = self.H
        dead = set(b4) | set(self.my_hole) | {self.my_disc, self.opp_disc}
        rv = np.array([c for c in range(52) if c not in dead], dtype=I64)
        if self.river_step > 1:
            rv = rv[self.rng.permutation(rv.size)[: max(8, rv.size // self.river_step)]]
        r, m = rv.size, h.shape[0]
        big = np.empty((r, m, 7), dtype=I64)
        big[:, :, 0] = h[:, 0]
        big[:, :, 1] = h[:, 1]
        big[:, :, 2:6] = np.array(b4, dtype=I64)
        big[:, :, 6] = rv[:, None]
        s = evalhands(big.reshape(-1, 7)).reshape(r, m)
        valid = (h[:, 0][None, :] != rv[:, None]) & (h[:, 1][None, :] != rv[:, None])
        mine7 = np.empty((r, 7), dtype=I64)
        mine7[:, 0:2] = np.array(self.my_hole, dtype=I64)
        mine7[:, 2:6] = np.array(b4, dtype=I64)
        mine7[:, 6] = rv
        mine = evalhands(mine7)
        win = ((mine[:, None] > s) + 0.5 * (mine[:, None] == s)) * valid
        nv = np.maximum(1, valid.sum(0))
        self.t_winp = win.sum(0) / nv
        sm = np.where(valid, s, -1)
        srt = np.sort(sm, axis=1)
        pct = np.empty((r, m))
        for i in range(r):
            row = srt[i]
            pct[i] = np.searchsorted(row, s[i], "left") + np.searchsorted(row, s[i], "right")
        ninv = m - valid.sum(1)
        pct = (pct * 0.5 - ninv[:, None]) / np.maximum(1, m - ninv)[:, None]
        self.t_str = (pct * valid).sum(0) / nv
        self.t_ready = True

    def ensure_river(self):
        if self.r_ready:
            return
        b5 = self.cur_board[:5]
        if not self.t_ready:
            self._filter(b5[3])
        self._filter(b5[4])
        h = self.H
        m = h.shape[0]
        s = evalhands(np.column_stack((h, np.broadcast_to(np.array(b5, dtype=I64), (m, 5)))))
        mine = int(evalhands(np.array([self.my_hole + b5], dtype=I64))[0])
        self.r_winp = (mine > s) + 0.5 * (mine == s)
        srt = np.sort(s)
        self.r_str = (np.searchsorted(srt, s, "left") + np.searchsorted(srt, s, "right")) * (0.5 / m)
        self.r_ready = True

    # ------------------------------------------------------------- pre-flop
    def pf_eq(self):
        return float(PF_E[self.my_bkt] @ self.wp / max(1e-12, self.wp.sum()))

    def pf_eq_top(self, frac, weights=None):
        wp = self.wp if weights is None else weights
        target = frac * float(wp.sum())
        acc = 0.0
        num = 0.0
        row = PF_E[self.my_bkt]
        for j in range(NB - 1, -1, -1):
            take = min(float(wp[j]), target - acc)
            if take <= 0:
                break
            num += take * float(row[j])
            acc += take
        return num / acc if acc > 0 else float(row[NB - 1])

    def preflop(self, req):
        c = req["to_call"]
        pot = req["pot"]
        legal = req["legal_actions"]
        can_raise = "raise" in legal
        mn = req.get("min_raise", 0)
        mx = req.get("max_raise", 0)
        p = self.my_pct
        st = self.opp
        x = req["opp_bet"]
        stack = req["my_stack"]

        if c == 0:                                   # big blind option after a limp
            thr = P["pf_iso"]
            fo = st.freq("p_vsopen")
            if st.n["p_vsopen"] >= 5 and fo[0] > 0.5:
                thr -= 0.20
            if can_raise and p >= thr:
                return "raise", max(mn, min(mx, 8))
            return "check", None

        if self.button and x == 2 and not req.get("history"):        # first to act
            fo = st.freq("p_vsopen")
            reraise = float(fo[2:].sum())
            shove = float(fo[4:].sum())
            thr = P["pf_open"]
            if reraise > 0.20:
                thr += (reraise - 0.20) * 1.2
            if st.n["p_vsopen"] >= 6 and fo[0] > 0.55:
                thr = 0.0
            if shove > 0.25:
                if self.pf_eq_top(min(1.0, shove + 0.05), np.ones(NB)) >= 0.47:
                    return ("raise", max(mn, min(mx, 5))) if can_raise else ("call", None)
                return "fold", None
            if p >= thr and can_raise:
                return "raise", max(mn, min(mx, 5))
            return "fold", None

        # facing a raise
        eq = self.pf_eq()
        need = c / float(pot + c)
        allin = c >= stack or x >= 80
        if allin:
            return ("call", None) if eq >= need + P["pf_allin_margin"] + 0.02 * self.risk else ("fold", None)
        r = (P["pf_r_ip"] if self.button else P["pf_r_oop"]) - P["pf_r_slope"] * max(0.0, 0.5 - p)
        if x >= 15:
            r -= 0.03
        if can_raise:
            eqt = self.pf_eq_top(0.5)
            if eqt >= P["pf_3bet_eq"] and p >= P["pf_3bet_p"]:
                to = int(round(x * (2.4 if self.button else 3.4)))
                to = max(mn, to)
                if to >= 70:
                    to = mx
                return "raise", min(mx, to)
            f3 = st.freq("p_vs3bet")
            if (not self.button and st.n["p_vs3bet"] >= 5 and f3[0] > 0.6 and x <= 8
                    and 0.30 <= p <= 0.55 and self.rng.random() < 0.4):
                return "raise", max(mn, min(mx, int(round(x * 3.4))))
        if eq * r >= need + P["pf_call_margin"]:
            return "call", None
        return "fold", None

    # ------------------------------------------------------------ post-flop
    @staticmethod
    def ev_bet(pot, b, fold, rho, wn, winp, q):
        cont = q >= fold
        wc = wn[cont]
        pc = winp[cont]
        mass = float(wc.sum())
        if mass <= 1e-9:
            return pot, 1.0
        val = pc * (pot + 2.0 * b) - b
        if rho > 0:
            rz = q[cont] >= 1.0 - rho
            wr = float(wc[rz].sum())
            if wr > 0 and float((wc[rz] * pc[rz]).sum()) / wr < 0.38:
                val = np.where(rz, -float(b), val)
        ev = (1.0 - mass) * pot + float((wc * val).sum())
        return ev, float((wc * pc).sum()) / mass

    def decide_post(self, req, w, winp, strength):
        street = req["street"]
        pot = float(req["pot"])
        c = req["to_call"]
        legal = req["legal_actions"]
        stack = req["my_stack"]
        mn = req.get("min_raise", 0)
        mx = req.get("max_raise", 0)
        rl = req.get("raises_left", 0)
        ip = self.button
        river = street == "river"
        can_raise = "raise" in legal
        st = self.opp
        tot = float(w.sum())
        if not tot > 0:
            w = np.ones_like(winp, dtype=float)
            tot = float(w.sum())
        wn = w / tot
        eq = float(wn @ winp)
        q = wpct(strength, w)
        rnd = float(self.rng.random())
        kc = min(2, self.rp.calls[1])
        mult = st.fold_mult(kc)
        if TRACE:
            TRACE("#%d %s %s | %s | pot %d call %d eq %.3f mult %.2f k%d" % (
                self.hand_id, street, " ".join(CARD_STR[x] for x in (self.my_hole or self.my3)),
                " ".join(CARD_STR[x] for x in self.cur_board), pot, c, eq, mult, kc))

        if c == 0:
            ev_check = eq * pot - (0.0 if ip else P["oop_pen"] * pot)
            best = None
            if can_raise:
                fr = {"flop": (0.33, 0.66, 1.0), "turn": (0.5, 0.75, 1.2)}.get(street, (0.4, 0.75, 1.25))
                cands = set()
                for f in fr:
                    cands.add(int(max(mn, min(mx, round(f * pot)))))
                if mx <= P["max_overbet"] * pot:
                    cands.add(int(mx))
                rho = float(st.freq("facing")[2:].sum()) if rl > 1 else 0.0
                options = []
                for b in sorted(cands):
                    fold = st.fold_frac(b / pot, kc)
                    ev, eqc = self.ev_bet(pot, b, fold, rho if b < mx else 0.0, wn, winp, q)
                    if TRACE:
                        TRACE("    bet %d fold %.2f ev %.1f eqc %.2f (check %.1f)" % (b, fold, ev, eqc, ev_check))
                    options.append((ev, b, eqc))
                    if best is None or ev > best[0] + 1e-9:
                        best = (ev, b, eqc)
                # A big turn bet with a weak hand is mostly a bluff; making it every time leaves the
                # big size bluff-heavy and readable.  Keep it only part of the time.
                if (street == "turn" and best is not None and best[1] >= 0.8 * pot
                        and eq < P["turn_bluff_eq"] and float(self.rng.random()) >= P["turn_thin"]):
                    smaller = [o for o in options if o[1] < 0.8 * pot]
                    best = max(smaller) if smaller else None
            if best is not None:
                ev, b, eqc = best
                if ev > ev_check + P["bet_margin"] * pot:
                    if eqc >= 0.5 or not river or rnd < 0.8:
                        return "raise", b
                if mult >= 0.7:
                    if river and eq < 0.28:
                        pb = ((P["bluff_ip"] if ip else P["bluff_oop"]) * min(2.5, mult * mult)
                              * (1.0 if eq < 0.18 else 0.5))
                        if rnd < pb:
                            return "raise", int(max(mn, min(mx, round(0.7 * pot))))
                    elif (not river and eq < 0.5 and ev > ev_check - P["semi_slack"] * pot
                          and rnd < P["semi_prob"]):
                        return "raise", b
            return "check", None

        # facing a bet or raise
        closes = river or c >= stack
        r = 1.0 if closes else (P["r_ip"] if ip else P["r_oop"])
        ev_call = eq * r * (pot + c) - c
        margin = 0.0
        if c > 40:
            margin = (P["big_call_margin"] * (2.0 if c > 100 else 1.0) * max(0.0, 1.0 + 0.4 * self.risk)
                      * (pot + c))
        if can_raise:
            x = req["opp_bet"]
            base = pot + c
            cands = set()
            for g in (0.6, 1.0):
                cands.add(int(max(mn, min(mx, x + round(g * base)))))
            if mx <= x + P["max_overbet"] * base:
                cands.add(int(mx))
            fr = st.freq("facing_raise")
            fold = float(fr[0])
            cont = q >= fold
            wc = wn[cont]
            mass = float(wc.sum())
            if mass > 1e-9:
                eqc = float((wc * winp[cont]).sum()) / mass
                best = None
                for to in sorted(cands):
                    inc = to - x
                    ev = (1.0 - mass) * pot + mass * (eqc * (pot + inc) - (1.0 - eqc) * (c + inc))
                    if eqc < 0.6:
                        ev -= 0.05 * (c + inc)
                    if best is None or ev > best[0] + 1e-9:
                        best = (ev, to)
                ev, to = best
                ref = max(ev_call, 0.0)
                if TRACE:
                    TRACE("    call ev %.1f | raise to %d ev %.1f eqc %.2f fold %.2f" % (ev_call, to, ev, eqc, fold))
                if eqc >= P["vraise_eqc"] and ev > ref + P["vraise_margin"] * base:
                    return "raise", to
                if (st.n["facing_raise"] >= 4 and fold >= 0.6 and ev > ref + 0.10 * base and rnd < 0.5):
                    return "raise", to
                if (not river and 0.30 <= eq <= 0.50 and eqc >= 0.30 and ev > ref and rnd < 0.10):
                    return "raise", to
        if ev_call > margin:
            return "call", None
        return "fold", None

    # ------------------------------------------------------------- requests
    def on_action(self, req):
        act, amt = self._choose(req)
        if self.cap is not None:
            act, amt = self._cap_action(req, act, amt)
        return act, amt

    def _cap_action(self, req, act, amt):
        """Keep this hand's total commitment within self.cap (see on_new_hand)."""
        legal = req.get("legal_actions", [])
        passive = ("check", None) if "check" in legal else ("fold", None)
        stack = req.get("my_stack")
        if stack is None:
            return passive
        room = self.cap - (self.start_stack - int(stack))      # chips we may still add
        c = int(req.get("to_call", 0))
        if act == "raise":
            if "raise" in legal and amt is not None:
                a = max(int(req["min_raise"]), min(int(req["max_raise"]), int(amt)))
                top = int(req.get("my_bet", 0)) + room
                if a <= top:
                    return "raise", a
                if top >= int(req["min_raise"]) and top >= 0.6 * a:
                    return "raise", top
            act = "call" if c > 0 else "check"
        if act == "call" and "call" in legal and c <= room:
            return "call", None
        return passive

    def _choose(self, req):
        self._budget(req)
        legal = req.get("legal_actions", [])
        if self.locked:
            return ("check", None) if "check" in legal else ("fold", None)
        street = req.get("street", "preflop")
        if self.time_left < 2500:                    # emergency: no computation at all
            if "check" in legal:
                return "check", None
            cheap = req.get("to_call", 0) <= 0.2 * req.get("pot", 0)
            return ("call", None) if cheap and self.my_pct > 0.5 else ("fold", None)
        self.cur_board = [cid(c) for c in req.get("board", [])]
        self.sync(req.get("history", []))
        if street == "preflop":
            return self.preflop(req)
        if street == "flop":
            self.ensure_sim(self.cur_board[:3])
            k = int(np.argmax(self.keep_equities()))
            return self.decide_post(req, self.sim["w"], self.sim["out"][k], self.sim["V"])
        if self.H is None:
            raise RuntimeError("no range after discard")
        if street == "turn":
            self.ensure_turn()
            return self.decide_post(req, self.W, self.t_winp, self.t_str)
        self.ensure_river()
        mode = int(P["solve_mode"])
        facing = req.get("to_call", 0) > 0
        if (SOLVE and req.get("pot", 0) >= P["solve_pot"]
                and (mode == 0 or (mode == 1 and not facing) or (mode == 2 and facing))):
            try:
                res = self.river_solve(req)
                if res is not None:
                    return res
            except Exception as exc:                    # fall back to the heuristic
                self.errors += 1
                if self.errors <= 20:
                    print("solver error: %r" % (exc,), file=sys.stderr)
        return self.decide_post(req, self.W, self.r_winp, self.r_str)

    def on_discard(self, req):
        self._budget(req)
        hole = [cid(c) for c in req.get("hole_cards", [])]
        if len(hole) == 3:
            self.my3 = hole
        flop = [cid(c) for c in req.get("board", [])][:3]
        self.cur_board = flop
        if self.time_left < 2500 and self.sim is None:
            self.flop = flop
            return CARD_STR[min(self.my3)]
        self.ensure_sim(flop)
        eqs = self.keep_equities()
        k = int(np.argmax(eqs))
        return CARD_STR[self.my3[2 - k]]

    def on_discard_result(self, msg):
        self.my_hole = [cid(c) for c in msg.get("hole_cards", [])]
        self.my_disc = cid(msg["my_discard"])
        self.opp_disc = cid(msg["opp_discard"])
        if self.flop is None:
            return
        self.init_range()

    @staticmethod
    def _observe(st, info):
        st.obs(info["ctx"], info["action"], info["cls"])
        if info["ctx"] == "facing" and info["street"] != "preflop":
            st.fold_exp[info["k"]] += fold_base(info["sf"])
            st.fold_obs[info["k"]] += 1.0 if info["action"] == "fold" else 0.0
        key = (info["street"], info["ctx"])
        if key in st.agg_n:
            st.agg_n[key] += 1.0
            st.agg_k[key] += 1.0 if info["action"] == "raise" else 0.0

    # ------------------------------------------------------------ river solver
    def my_range_river(self, history):
        """Our possible holdings on the river, weighted as the opponent could infer them."""
        b5 = self.cur_board[:5]
        flop = b5[:3]
        dead = set(b5) | {self.my_disc, self.opp_disc}
        h = pairs_of([c for c in range(52) if c not in dead])
        m = h.shape[0]
        rp = Replay(self.button)
        specs = []
        for e in history:
            info = rp.step(e)
            if info["who"] == 0:
                sp = self.make_spec(info, self.me)
                if sp is not None:
                    specs.append((info["street"], sp))
        wp = np.ones(NB)
        for street, sp in specs:
            if street == "preflop":
                wp = wp * lik(sp, wpct(BIN_C, wp))
                wp = wp / max(1e-12, wp.max())
        d = self.my_disc
        dcol = np.full(m, d, dtype=I64)
        tri = np.sort(np.column_stack((h, dcol)), axis=1)
        w = wp[PF_BKT[tri[:, 0] + C2T[tri[:, 1]] + C3T[tri[:, 2]]]]
        if self.ref is None:
            self.ref = flop_ref(flop)
        fl = np.broadcast_to(np.array(flop, dtype=I64), (m, 3))
        vh = flop_value(np.column_stack((h, fl)), self.ref)
        va = flop_value(np.column_stack((h[:, 0], dcol, fl)), self.ref)
        vb = flop_value(np.column_stack((h[:, 1], dcol, fl)), self.ref)
        w = w * (0.05 + 0.9 * sig((vh - np.maximum(va, vb)) / 0.03))      # we keep the best pair
        st = {"flop": vh,
              "turn": evalhands(np.column_stack((h, np.broadcast_to(np.array(b5[:4], dtype=I64), (m, 4))))),
              "river": evalhands(np.column_stack((h, np.broadcast_to(np.array(b5, dtype=I64), (m, 5)))))}
        for street, sp in specs:
            if street in st:
                w = w * lik(sp, wpct(st[street], w))
                w = w / max(1e-12, w.max())
        return h, w, st["river"]

    @staticmethod
    def _prune(h, w, s, cap, keep=None):
        idx = np.nonzero(w >= 0.004 * w.max())[0]
        if idx.size > cap:
            idx = idx[np.argsort(-w[idx])[:cap]]
        if keep is not None and keep not in set(idx.tolist()):
            idx = np.append(idx, keep)
        return h[idx], w[idx], s[idx], idx

    def river_solve(self, req):
        """Pick our river action with the range-vs-range solver; None to fall back."""
        left = max(1, self.total_hands - self.hand_id + 1)
        per = self.time_left / left
        if per < 45 or self.time_left < 8000:
            return None
        iters, budget = (300, 0.45) if per > 110 else ((180, 0.25) if per > 70 else (100, 0.12))
        hm, wm, sm = self.my_range_river(req.get("history", []))
        lo, hi = min(self.my_hole), max(self.my_hole)
        im = np.nonzero((hm[:, 0] == lo) & (hm[:, 1] == hi))[0]
        if im.size != 1:
            return None
        hm, wm, sm, idx = self._prune(hm, wm, sm, int(P["solve_cap"]), int(im[0]))
        me_i = int(np.nonzero(idx == im[0])[0][0])
        b5 = self.cur_board[:5]
        so = evalhands(np.column_stack((self.H, np.broadcast_to(np.array(b5, dtype=I64), (self.H.shape[0], 5)))))
        ho, wo, so, _ = self._prune(self.H, self.W, so, int(P["solve_cap"]))
        pot = req["pot"]
        mb, ob = req.get("my_bet", 0), req.get("opp_bet", 0)
        base = (pot - mb - ob) / 2.0
        river_acts = [e for e in req.get("history", []) if e.get("street") == "river"]
        acted = [any(e.get("actor") == "me" for e in river_acts),
                 any(e.get("actor") == "opp" for e in river_acts)]
        can_raise = "raise" in req["legal_actions"]
        last = max(2, int(req.get("min_raise", ob + 2)) - ob) if can_raise else 2
        rl = int(req.get("raises_left", 0)) if can_raise else 0
        sv = RiverSolver((hm, ho), (wm, wo), (sm, so), base, [mb, ob],
                         [req["my_stack"], req["opp_stack"]], acted, min(rl, 3), last)
        done = sv.solve(iters, time.monotonic() + budget)
        acts, probs = sv.root_strategy(me_i)
        probs = np.where(probs < P["purify"], 0.0, probs)
        if probs.sum() <= 0:
            return None
        probs = probs / probs.sum()
        a = int(self.rng.choice(len(acts), p=probs))
        self.solves += 1
        if TRACE:
            TRACE("    solver %dx%d it %d: %s" % (len(wm), len(wo), done, " ".join(
                "%s%s:%.2f" % (x[0], "" if x[1] is None else x[1], pr) for x, pr in zip(acts, probs))))
        return acts[a]

    # ------------------------------------------------------------ hand end
    def on_result(self, msg):
        net = int(msg.get("net", 0))
        self.net += net
        if self.cap is not None:
            self.capped_net += net
        elif not self.locked:
            self.free_n += 1
            self.free_s1 += net
            self.free_s2 += float(net) * net
        st = self.opp
        hist = msg.get("history", []) or []
        rp = Replay(self.button)
        last_agg = None
        last_s = 0.0
        sz_bets = []
        for e in hist:
            info = rp.step(e)
            if info["who"] != 1:
                self._observe(self.me, info)
                continue
            st.obs(info["ctx"], info["action"], info["cls"])
            if info["ctx"] == "facing" and info["street"] != "preflop":
                st.fold_exp[info["k"]] += fold_base(info["sf"])
                st.fold_obs[info["k"]] += 1.0 if info["action"] == "fold" else 0.0
            key = (info["street"], info["ctx"])
            if key in st.agg_n:
                st.agg_n[key] += 1.0
                st.agg_k[key] += 1.0 if info["action"] == "raise" else 0.0
            if info["action"] == "raise" and info["street"] in ("turn", "river"):
                last_agg = True
                last_s = info.get("s", 0.0)
            if info["action"] == "raise" and info["street"] in ("flop", "turn", "river") and "s" in info:
                sz_bets.append((info["street"], 0 if info["cls"] == 0 else (1 if info["cls"] == 1 else 2)))
        oh = msg.get("opp_hole_cards")
        od = msg.get("opp_discard")
        board = msg.get("board") or []
        if msg.get("result") == "showdown" and oh and od and len(board) == 5:
            b5 = [cid(x) for x in board]
            o2 = [cid(x) for x in oh]
            d = cid(od)
            flop = b5[:3]
            ref = flop_ref(flop)
            fl = np.array(flop, dtype=I64)
            trio = [o2, [o2[0], d], [o2[1], d]]
            vals = flop_value(np.array([t + flop for t in trio], dtype=I64), ref)
            eps = 0.12
            l1 = 1.0 - eps if vals[0] >= vals[1:].max() - 0.01 else eps / 2
            l2 = 1.0 - eps if (d >> 2) <= min(o2[0] >> 2, o2[1] >> 2) else eps / 2
            pi = st.pi * np.array([l1, l2, 1.0 / 3.0])
            pi = np.maximum(pi / pi.sum(), 0.02)
            st.pi = pi / pi.sum()
            nb = {"flop": 3, "turn": 4, "river": 5}
            for street_, g in sz_bets:
                bb = b5[:nb[street_]]
                bs_ = set(bb)
                hh = pairs_of([x for x in range(52) if x not in bs_ and x not in o2])
                ref_ = np.sort(evalhands(np.column_stack((hh, np.broadcast_to(np.array(bb, dtype=I64), (hh.shape[0], len(bb)))))))
                v_ = int(evalhands(np.array([o2 + bb], dtype=I64))[0])
                st.sz_n[g] += 1.0
                pc_ = (np.searchsorted(ref_, v_, "left") + np.searchsorted(ref_, v_, "right")) * 0.5 / ref_.size
                st.sz_sum[g] += pc_
                st.sz_sq[g] += pc_ * pc_
            if last_agg is not None:
                bs = set(b5)
                rest = [x for x in range(52) if x not in bs]
                h = pairs_of(rest)
                s = np.sort(evalhands(np.column_stack((h, np.broadcast_to(np.array(b5, dtype=I64), (h.shape[0], 5))))))
                ov = int(evalhands(np.array([o2 + b5], dtype=I64))[0])
                pct = (np.searchsorted(s, ov, "left") + np.searchsorted(s, ov, "right")) * 0.5 / s.size
                st.bluff_n += 1.0
                if pct < P["bluff_pct"]:
                    st.bluff_k += 1.0
                if last_s >= P["big_s"]:
                    st.bigb_n += 1.0
                    if pct < P["bluff_pct"]:
                        st.bigb_k += 1.0


# ---------------------------------------------------------------------- main

def legalize(req, act, amt):
    legal = req.get("legal_actions", [])
    if act == "raise":
        if "raise" in legal and amt is not None:
            a = int(amt)
            a = max(int(req["min_raise"]), min(int(req["max_raise"]), a))
            return {"type": "action", "action": "raise", "amount": a}
        act = "call" if "call" in legal else "check"
    if act == "call" and "call" not in legal:
        act = "check"
    if act == "check" and "check" not in legal:
        act = "fold"
    if act == "fold" and "fold" not in legal:
        act = "check"
    if act not in legal and legal:
        act = "check" if "check" in legal else ("fold" if "fold" in legal else legal[0])
    return {"type": "action", "action": act}


def safe_action(req):
    legal = req.get("legal_actions", [])
    if "check" in legal:
        return {"type": "action", "action": "check"}
    c = req.get("to_call", 0)
    if "call" in legal and c <= 0.25 * max(1, req.get("pot", 0)):
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else "call"}


def main():
    bot = Bot()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        reply = None
        try:
            if t == "action_request":
                try:
                    act, amt = bot.on_action(msg)
                    reply = legalize(msg, act, amt)
                except Exception as exc:                    # never crash, never stall
                    bot.errors += 1
                    if bot.errors <= 20:
                        print("action error: %r" % (exc,), file=sys.stderr)
                    reply = safe_action(msg)
            elif t == "discard_request":
                try:
                    card = bot.on_discard(msg)
                    if card not in msg.get("hole_cards", []):
                        raise ValueError("bad discard")
                except Exception as exc:
                    bot.errors += 1
                    if bot.errors <= 20:
                        print("discard error: %r" % (exc,), file=sys.stderr)
                    cards = msg.get("hole_cards", [])
                    card = min(cards, key=cid) if cards else "2s"
                reply = {"type": "discard", "card": card}
            elif t == "new_hand":
                bot.on_new_hand(msg)
            elif t == "discard_result":
                bot.on_discard_result(msg)
            elif t == "hand_result":
                bot.on_result(msg)
            elif t == "init":
                bot.on_init(msg)
            elif t == "match_end":
                break
        except Exception as exc:
            bot.errors += 1
            if bot.errors <= 20:
                print("state error: %r" % (exc,), file=sys.stderr)
        if reply is not None:
            out.write(json.dumps(reply, separators=(",", ":")) + "\n")
            out.flush()
    if DEBUG:
        st = bot.opp
        print("net %d errors %d solves %d fold_mult %.2f beta %.2f pi %s" % (
            bot.net, bot.errors, bot.solves, st.fold_mult(0), st.beta("river"), np.round(st.pi, 2)), file=sys.stderr)


if __name__ == "__main__":
    main()
