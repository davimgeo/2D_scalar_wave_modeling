#!/bin/bash

Nx=416
Nz=160
Nt=5001

dx=10
dz=10
dt=0.0004
fmax=10

spread=52
spacing=5

depth_src=8
depth_rec=0

interfaces=(40 100) 
vp_interfaces=(1500 2000 2500)

snapshots=1			# Habilitar snapshots(1=True)
snap_num=100

################ Creating a temporary file to store the parameters ################
printf '%s ' $Nx $Nz $Nt $dx $dz $fmax $spread $spacing $depth_src \
$depth_rec $snapshots $snap_num > temp_par.txt
printf '%s\n' "1a" "$(printf "%s " "$dt")" . x | ex temp_par.txt
printf '%s\n' "1a" "$(printf "%s " "${interfaces[@]}")" . x | ex temp_par.txt
printf '%s\n' "1a" "$(printf "%s " "${vp_interfaces[@]}")" . x | ex temp_par.txt

python3 ./bin/OOP_wave.py

rm temp_par.txt
