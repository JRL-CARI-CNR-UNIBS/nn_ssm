close all; clear all; clc;

% Load dataset
cd(fileparts(which(mfilename)));

file_name = "ssm_dataset_scaling_simple_500k";

if isfile(file_name+".mat")
    disp("Loading dataset..")
    load(file_name+".mat");
else
    disp("Saving dataset as .mat file..")
    [input,output] = readBinDistance(file_name);
    save(file_name+".mat","input","output");
end

disp("Ready")

max_epochs = 5000;
hiddenLayerSize = [50,50,50];
ws_save = "nn_scaling_simple_"+string(mat2str(hiddenLayerSize))+".mat";

% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by Neural Fitting app
% Created 14-May-2023 12:56:58
%
% This script assumes these variables are defined:
%
%   input - input data.
%   output - target data.

perc = 0.8;

size_dataset = int32(perc*size(input,2));
x = input(:,1:size_dataset);
t = output(:,1:size_dataset);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
net=feedforwardnet(hiddenLayerSize,trainFcn);

% Set last activation as Sigmoid
net.layers{length(net.layers)}.transferFcn = 'tansig';

%view(net)

% hiddenLayerSize = 10;
% net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainParam.epochs = max_epochs;
net.trainParam.max_fail = 10;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,x,t,'useGPU','yes');


% [net,tr] = train(net,x,t,'useGPU','yes');
% [net,tr] = train(net,x,t,'useParallel','yes','useGPU','only','showResources','yes');

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotregression(t,y)
figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end

save(ws_save);
