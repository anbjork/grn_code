function simulate(input_file)

addpath(genpath('genespider'));
maxNumCompThreads(1);

% Matlab is bananas and has a default seed set at startup :facepalm:
% Btw, R does too, and that fact makes my brain hurt. Why, why?
% Such a broken world..
rng("shuffle")

% Load full JSON dictionary from the provided input file
fid = fopen(input_file, 'r');
if fid == -1
    error('Could not open JSON file: %s', input_file);
end
raw = fread(fid, inf, 'char=>char')';
fclose(fid);
job_specifications = jsondecode(raw);

for i = 1:numel(job_specifications)

    job_specification = job_specifications(i);

    param_tag = job_specification.parameter_tag;
    fprintf('Matlab, simulating: %s\n', param_tag);

    parameters = job_specification.parameters;
	negbin_prob = parameters.negbin_prob;
    disper = parameters.dispersion;
    cell_count = parameters.cell_count;
    snr = parameters.snr;

	globs = job_specification.python_global_parameters_for_matlab;	
	number_of_genes = globs.number_of_genes;
    average_network_degree = globs.average_network_degree;
	

    % Generate GRN
    A = datastruct.large_scalefree(number_of_genes, average_network_degree);
    P = -repmat(eye(number_of_genes), 1, cell_count);

    [Y, X, Ed, Eg, SCC] = datastruct.scdata(...
        A, P, ...
        "SNR", snr, ...
        'SNR_model', "SNR_vov", ...
        'negbin_prob', negbin_prob, ...
        'disper', disper, ...
        'raw_counts', true);

    % Matrix export
    names = job_specification.simulation_matrix_names;
    values = {A, Y, X, P, SCC, Ed, Eg};

    matrix_files = job_specification.simulation_matrix_files;
    file_dict = struct();
    for idx = 1:numel(names)
        name = names{idx};
        if isfield(matrix_files, name)
            filename = matrix_files.(name);
        else
            error('Missing filename for %s in JSON input.', name);
        end

        data = values{idx};

        if exist(filename, 'file') == 2
            delete(filename);
        end
        h5create(filename, '/data', size(data));
        h5write(filename, '/data', data);

        file_dict.(name) = filename;
    end	
	% Make flag file to indicate completion
	flag_file = job_specification.simulation_completed_flag_file;
	fid = fopen(flag_file, 'w');
	if fid == -1
		error('Could not create flag file: %s', flag_file);
	end
	fclose(fid);
end
end
