## Scalabilite forte : Poisson 3D 512^3 (134 M d'inconnues), 1140 iterations

| GPU | aware-overlap : temps (s), acceleration, efficacite | aware-sync : temps (s), acceleration, efficacite | staged-sync : temps (s), acceleration, efficacite |
|---|---|---|---|
| 1 | 9.240 ; 1.0x ; 100 % | 9.224 ; 1.0x ; 100 % | 9.278 ; 1.0x ; 100 % |
| 2 | 4.693 ; 2.0x ; 98 % | 4.707 ; 2.0x ; 98 % | 4.829 ; 1.9x ; 96 % |
| 4 | 2.434 ; 3.8x ; 95 % | 2.450 ; 3.8x ; 94 % | 2.967 ; 3.1x ; 78 % |
| 8 | 1.276 ; 7.2x ; 90 % | 1.349 ; 6.8x ; 85 % | 1.815 ; 5.1x ; 64 % |
| 16 | 0.698 ; 13.2x ; 83 % | 0.795 ; 11.6x ; 72 % | 1.253 ; 7.4x ; 46 % |

## Scalabilite faible : 256 x 256 x 256P (16,8 M d'inconnues par GPU), temps par iteration

| GPU | aware-overlap : us/iter, efficacite | aware-sync : us/iter, efficacite | staged-sync : us/iter, efficacite |
|---|---|---|---|
| 1 | 1048 ; 100 % | 1043 ; 100 % | 1063 ; 100 % |
| 2 | 1060 ; 99 % | 1066 ; 98 % | 1100 ; 97 % |
| 4 | 1086 ; 97 % | 1096 ; 95 % | 1214 ; 88 % |
| 8 | 1093 ; 96 % | 1103 ; 95 % | 1221 ; 87 % |
| 16 | 1091 ; 96 % | 1110 ; 94 % | 1214 ; 88 % |

## Ou part le temps (scalabilite forte, rang le plus lent) : attente du halo et Allreduce (s)

| GPU | aware-overlap : halo / allreduce / total | aware-sync : halo / allreduce / total | staged-sync : halo / allreduce / total |
|---|---|---|---|
| 1 | 0.002 / 0.008 / 9.240 | 0.000 / 0.008 / 9.224 | 0.000 / 0.008 / 9.278 |
| 2 | 0.043 / 0.032 / 4.693 | 0.035 / 0.027 / 4.707 | 0.106 / 0.022 / 4.829 |
| 4 | 0.053 / 0.070 / 2.434 | 0.044 / 0.063 / 2.450 | 0.482 / 0.172 / 2.967 |
| 8 | 0.119 / 0.059 / 1.276 | 0.113 / 0.103 / 1.349 | 0.482 / 0.166 / 1.815 |
| 16 | 0.120 / 0.043 / 0.698 | 0.112 / 0.114 / 0.795 | 0.488 / 0.183 / 1.253 |
