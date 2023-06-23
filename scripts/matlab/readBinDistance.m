function [input,output] = readBinDistance(file_name)

file_name = file_name+".bin";
fileId = fopen(file_name);

parent = [];
child = [];
obs = [];
distance = [];

while ~feof(fileId)
    tmp_parent = fread(fileId,6,'double');
    tmp_child = fread(fileId,6,'double');
    tmp_obs = fread(fileId,3,'double');
    tmp_distance = fread(fileId,1,'double');

    parent = [parent,tmp_parent];
    child = [child,tmp_child];
    obs = [obs,tmp_obs];

    distance = [distance,tmp_distance];
end

input = [parent;child;obs];
output = distance;
end