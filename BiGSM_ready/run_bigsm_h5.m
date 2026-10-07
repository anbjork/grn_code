function run_bigsm_h5(in_path, out_path)

    Y = h5read(in_path, '/Y');
    P = h5read(in_path, '/P');
    max_iter = h5read(in_path, '/max_iter');

    N = size(Y, 1);

    A = bigsm(Y, P, double(max_iter), [N N]);

    h5create(out_path, '/A', size(A));
    h5write(out_path, '/A', A);

end