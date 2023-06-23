function [input,output] = readBinScaling(file_name)

file_name = file_name+".bin";
fileId = fopen(file_name);

parent = [];
child = [];
obs = [];
scaling = [];

while ~feof(fileId)
    tmp_parent = fread(fileId,6,'double');
    tmp_child = fread(fileId,6,'double');
    tmp_obs = fread(fileId,3,'double');
    tmp_length = fread(fileId,1,'double');
    tmp_poi = fread(fileId,9,'double');
    tmp_dq = fread(fileId,6,'double');
    tmp_v_safe = fread(fileId,3,'double');
    tmp_speed = fread(fileId,3,'double');
    tmp_distance = fread(fileId,3,'double');
    tmp_scaling = fread(fileId,3,'double');
    tmp_average_scaling = fread(fileId,1,'double');


    parent = [parent,tmp_parent];
    child = [child,tmp_child];
    obs = [obs,tmp_obs];

    scaling = [scaling,tmp_average_scaling];
end

input = [parent;child;obs];
output = scaling;

end