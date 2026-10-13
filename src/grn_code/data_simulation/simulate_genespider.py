
import json
from pathlib import Path
import anton_util
import subprocess
import datetime
from inspect import cleandoc
import datetime
import time


def block_print(s):
    return ' '.join(cleandoc(s).splitlines())



def prepare_chunk(chunk, chunk_dir):

    import uuid
    chunk_file = chunk_dir / str(uuid.uuid4())
    with open(chunk_file, 'w') as f:
        json.dump(chunk, f)

    from grn_code.paths_anchor import package_root

    genespider_dir = Path(package_root).parents[1] / "genespider"
    sim_dir = Path(package_root) / "data_simulation"

    matlab_cmd = f"addpath('{sim_dir}'); addpath(genpath('{genespider_dir}')); simulate('{chunk_file}')"
    cmd = ["matlab", "-batch", matlab_cmd]

    anton_util.log_timestamp(' '.join(cmd))  # Nice for debugging Matlab separately

    return(cmd)


def terminate_process(process):
    process.terminate()
    s = f'Process {process.pid} exceeded job timeout of {job_timeout}'
    anton_util.log_timestamp(s)
    try:
        process.wait(timeout=termination_cleanup_timeout)
        s = block_print(f"""
                Process {process.pid} terminated gracefully 
                within {termination_cleanup_timeout} s timeout
                """)
        anton_util.log_timestamp(s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        s = (
            f'Process {process.pid} did not terminate gracefully within'
            '{termination_cleanup_timeout} s timeout, and was killed'
            )
        anton_util.log_timestamp(s)


def check_for_astronomical_results(job_specifications):

    # # Debug
    # mock_success = True
    # if mock_success:
    #     for job in job_specifications:
    #         job['simulation_succeeded'] = True
    #     return []
    # else:
    #     return job_specifications

    import h5py
    import pandas as pd

    astronomical_counts_cutoff = 1e9

    to_redo = []
    for job in job_specifications:

        simulation_files = job['simulation_matrix_files']
        with h5py.File(simulation_files['Y']) as f: 
            raw = f['data'][:] # pyright: ignore[reportIndexIssue]
        simulated_gene_names = [f'G{ii}' for ii in range(raw.shape[1])]  # pyright: ignore[reportAttributeAccessIssue]
        Y = pd.DataFrame(
            data = raw,
            columns = simulated_gene_names  # pyright: ignore[reportArgumentType]
            )
        mean_Y = Y.mean().mean()  # pyright: ignore[reportAttributeAccessIssue]

        if mean_Y > astronomical_counts_cutoff:
            # print(f"Simulation astronomical, redoing")
            to_redo.append(job)
        else:
            job['simulation_succeeded'] = True

    s = block_print(f"""
          {len(to_redo)} out of {len(job_specifications)} simulations
          were astronomical and will be rerun
          """)
    anton_util.log_timestamp(s)

    return to_redo


def split_on_retries(job_specifications):
    given_up = []
    retry = []
    for job in job_specifications:
        if job['simulation_retries'] < max_retries:
            retry.append(job)
        else:
            given_up.append(job)
    return given_up, retry



def job_manager(output_path, job_specifications):

    anton_util.log_timestamp('Script start')

    print(f'Chunk size: {chunk_size}')
    print(f'Max workers: {max_workers}')
    print(f'Job timeout: {job_timeout}')
    print(f'Termination cleanup timeout: {termination_cleanup_timeout}')

    # job_specifications = job_specifications[:5]  # Debug

    for job in job_specifications:
        job['simulation_retries'] = 0

    chunk_dir = Path(f'{output_path}/simulation/tmp_simulation_input_chunks')
    chunk_dir.mkdir(exist_ok = True, parents = True)

    processes = {}
    start_times = {}
    time_taken = {}
    wip = {}
    timeouts = {}

    job_queue = []
    for job_specification in job_specifications:
        f = job_specification['simulation_completed_flag_file']
        if not Path(f).exists():
            job_queue.append(job_specification)
    anton_util.log_timestamp(
            f'Found {len(job_queue)} not yet started jobs, '
            f'out of {len(job_specifications)} total jobs'
            )

    def requeue(jobs):
        given_up, retry = split_on_retries(jobs)
        for job in given_up:
            job['simulation_succeeded'] = False
        for job in retry:
            job['simulation_retries'] += 1
        job_queue.extend(retry)
        anton_util.log_timestamp(
                f'gave up on {len(given_up)} jobs, requeuing {len(retry)}')

    # job_queue = job_queue[ : 5]

    while True:

        while len(processes) < max_workers:
            if len(job_queue) == 0:
                break

            chunk = job_queue[-chunk_size : ]
            job_queue = job_queue[ : -chunk_size]
            cmd = prepare_chunk(chunk = chunk, chunk_dir = chunk_dir)
            # cmd = ['echo', 'mocked successful run']  # Debug

            p = subprocess.Popen(cmd)
            pid = p.pid

            processes[pid] = p
            start_times[pid] = datetime.datetime.now()
            wip[pid] = chunk
            anton_util.log_timestamp(f'Started process {pid}')

        # Optional: Sleep before polling, to catch immediately returning processes
        # in the same update iteration. So far mainly useful when debugging
        # with mocked jobs that return fast.
        time.sleep(1)

        for pid in list(processes.keys()):
            process = processes[pid]
            status = process.poll()
            anton_util.log_timestamp(f'Process {pid} status: {status}')
            timed_out = datetime.datetime.now() - start_times[pid] > job_timeout
            if status is None:
                if not timed_out:
                    anton_util.log_timestamp(
                            f'Process {pid} is still running within timeout'
                            )
                    continue
                elif timed_out:
                    terminate_process(process)
                    requeue(wip.pop(pid))
                    timeouts[pid] = True
            else:
                timeouts[pid] = False
                anton_util.log_timestamp(
                    f'Process {pid} returned with {status}'
                    )
                time_taken[pid] = datetime.datetime.now() - start_times[pid]
                anton_util.log_timestamp(
                    f'Process {pid} took {time_taken[pid]} to complete'
                    )
                if status != 0:
                    requeue(wip.pop(pid))
                else:
                    to_redo = check_for_astronomical_results(wip.pop(pid))
                    requeue(to_redo)
            processes.pop(pid)

        if len(processes) == 0 and len(job_queue) == 0:
            anton_util.log_timestamp('All simulations completed. Exiting job manager.')
            break

        anton_util.log_timestamp(f'Current active processes: {len(processes)}')
        anton_util.log_timestamp(f'Current jobs in queue: {len(job_queue)}')
        # I cannot be fucked waiting for logs to finish buffering
        print('', flush=True)

        time.sleep(10)


    run_metadata = {
        'timeouts': timeouts,
        'chunk_size': chunk_size,
        'start_times': start_times,
        'time_taken': time_taken,
        'wip': wip,
        'job_timeout': job_timeout.total_seconds(),
        'termination_cleanup_timeout': termination_cleanup_timeout,
        }
    anton_util.pickle_object(
            run_metadata,
            f'{output_path}/simulation/simulation_run_metadata.pkl'
            )

    tmp = [job['simulation_succeeded'] for job in job_specifications]
    anton_util.log_timestamp(f'Successful jobs: {sum(tmp)}')
    anton_util.log_timestamp(f'Failed jobs:     {len(job_specifications) - sum(tmp)}')
    anton_util.log_timestamp('Simulations finished')

    # returns the updated job specifications, including the simulation_retries 
    # count for each job. It would have been cleaner to return a separate
    # object instead of duplicating and modifying the input. But that would have
    # required adding simulation IDs to link against, and I chose to not
    # do that this time. Can consider for the future. IDs probably scale better
    # when adding things.
    return job_specifications



chunk_size = 1
max_workers = 10
max_attempts = 5
job_timeout = datetime.timedelta(minutes = 1)
termination_cleanup_timeout = 10

max_retries = max_attempts - 1

def main(output_path, job_specifications):

    updated_jobs_specs = job_manager(
            output_path = output_path, job_specifications = job_specifications)
    return updated_jobs_specs




