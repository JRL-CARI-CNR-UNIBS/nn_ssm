clc; close all; clear all;

I = 15;
O = 29;
Ntrn = 1000000;
n_H_layers = 3;

syms n_neurons
Ntrneq = Ntrn*O;
Nw = I*n_neurons+n_neurons+((n_H_layers-1)*(n_neurons*n_neurons+n_neurons))+n_neurons*O+O; 
Neq = Ntrn*O;

res = double(solve(Nw==Neq))
