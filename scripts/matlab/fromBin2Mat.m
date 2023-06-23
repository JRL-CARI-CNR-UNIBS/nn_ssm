clc; clear all; close all;

cd(fileparts(which(mfilename)));

file_name = "ssm_dataset_connection_500k";

if isfile(file_name+".mat")
    disp("Loading dataset..")
    load(file_name+".mat");
else
    disp("Saving dataset as .mat file..")
    [input,output] = readBinScaling(file_name);
    save(file_name+".mat","input","output");
end

disp("Ready")