
import json
from pathlib import Path
import anton_util
import subprocess
import datetime
from inspect import cleandoc
import datetime

anton_util.log_timestamp('Script start')


chunk_size = 1
max_workers = 10
job_timeout = datetime.timedelta(minutes = 1)
termination_cleanup_timeout = 10
print(f'Chunk size: {chunk_size}')
print(f'Max workers: {max_workers}')
print(f'Job timeout: {job_timeout}')
print(f'Termination cleanup timeout: {termination_cleanup_timeout}')

job_specifications = anton_util.unpickle_object(
    'outputs/simulation_specifications.pkl'
        )

chunk_dir = Path('outputs/simulation/tmp_simulation_input_chunks')
chunk_dir.mkdir(exist_ok = True, parents = True)


processes = {}
start_times = {}
time_taken = {}
wip = {}
timeouts = {}


def block_print(s):
    return ' '.join(cleandoc(s).splitlines())



def prepare_chunk(chunk):

    import uuid
    chunk_file = chunk_dir / str(uuid.uuid4())
    with open(chunk_file, 'w') as f:
        json.dump(chunk, f)
    cmd = ["matlab", "-batch", f"simulate('{chunk_file}')"]
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

    s = block_print(f"""
          {len(to_redo)} out of {len(job_specifications)} simulations
          were astronomical and will be rerun
          """)
    anton_util.log_timestamp(s)
    return to_redo




def job_manager():

    jobs_not_started = []
    for job_specification in job_specifications:
        f = job_specification['simulation_completed_flag_file']
        if not Path(f).exists():
            jobs_not_started.append(job_specification)
    anton_util.log_timestamp(
            f'Found {len(jobs_not_started)} not yet started jobs, '
            f'out of {len(job_specifications)} total jobs'
            )

    # jobs_not_started = jobs_not_started[ : 5]

    while True:

        while len(processes) < max_workers:
            if len(jobs_not_started) == 0:
                break

            chunk = jobs_not_started[-chunk_size : ]
            jobs_not_started = jobs_not_started[ : -chunk_size]
            cmd = prepare_chunk(chunk = chunk)
            p = subprocess.Popen(cmd)
            pid = p.pid

            processes[pid] = p
            start_times[pid] = datetime.datetime.now()
            wip[pid] = chunk
            anton_util.log_timestamp(f'Started process {pid}')

        for pid in list(processes.keys()):
            # anton_util.log_timestamp(f'Checking process {pid}')
            process = processes[pid]
            status = process.poll()
            anton_util.log_timestamp(f'Process {pid} status: {status}')
            if status is not None:
                anton_util.log_timestamp(
                    f'Process {pid} returned with {status}'
                    )
                timeouts[pid] = False
                process.wait()
                if status != 0:
                    jobs_not_started.extend(wip.pop(pid))
                else:
                    to_redo = check_for_astronomical_results(
                            wip[pid]
                            )
                    jobs_not_started.extend(to_redo)
            elif datetime.datetime.now() - start_times[pid] > job_timeout:
                terminate_process(process)
                jobs_not_started.extend(wip.pop(pid))
                timeouts[pid] = True
            else:
                anton_util.log_timestamp(
                        f'Process {pid} is still running within timeout'
                        )
                continue
            # Note the continue statement in the else block above.
            # Because of that, I can put code common to the
            # other cases here. I remember failing to notice that once.
            # It's not the most obvious logic, but does save
            # procomputing 2 conditions and then checking where to go,
            # and/or nesting control structures.
            # Also not seeing any obvious condition inversion that
            # wouldn't complicate things conceptually. Might be some,
            # haven't spent that much time trying
            time_taken[pid] = datetime.datetime.now() - start_times[pid]
            anton_util.log_timestamp(
                    f'Process {pid} took {time_taken[pid]} to complete'
                    )
            processes.pop(pid)

        if len(processes) == 0 and len(jobs_not_started) == 0:
            anton_util.log_timestamp('All simulations completed. Exiting job manager.')
            break

        anton_util.log_timestamp(f'Current active processes: {len(processes)}')
        anton_util.log_timestamp(f'Current jobs not started: {len(jobs_not_started)}')
        # I cannot be fucked waiting for logs to finish buffering
        print('', flush=True)

        import time
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
            'outputs/simulation/simulation_run_metadata.pkl'
            )
    return 0


def main():
    job_manager()

    anton_util.log_timestamp('Initial parameter grid simulated')

main()



