# submit a job to the appropriate cluster
submit_job = function(grid, script, allocation,
                      system = c('argo', 'sockeye'),
                      job_loop = 1,
                      jobs_per_array = 100) {
  system = match.arg(system)
  
  if (system == 'sockeye') {
    script %<>% gsub("\\.sh$", ".torque.sh", .)
    n_jobs = ceiling(nrow(grid)/ job_loop)
    # avoid a torque bug
    if (n_jobs == 1) {
      n_jobs = 2
    }
    ## Sockeye only lets you run 1,000 jobs at a time
    n_submissions = ifelse(n_jobs > jobs_per_array,
                           ceiling(n_jobs / jobs_per_array), 1)
    for (submission_idx in seq_len(n_submissions)) {
      job_start = (submission_idx - 1) * jobs_per_array + 1
      job_end = ifelse(submission_idx == n_submissions,
                       ifelse(n_jobs %% jobs_per_array == 0,
                              submission_idx * jobs_per_array,
                              job_start - 1 + n_jobs %% jobs_per_array),
                       submission_idx * jobs_per_array)
      system(paste0("qsub -A ", allocation, " -J ", job_start, "-", job_end,
                    " -v ", shQuote(paste0("JOB_SIZE=", job_loop)), " ", script))
    }
  } else if (system == 'argo') {
    n_jobs = ceiling(nrow(grid)/ job_loop)
    system(paste0("sbatch --array=1-", n_jobs, " ", script, " ", job_loop))
  }
}
