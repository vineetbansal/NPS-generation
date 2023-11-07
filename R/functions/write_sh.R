write_sh = function(job_name,
                    sh_file,
                    grid_file,
                    inner_file,
                    system = c('sockeye', 'argo'),
                    env = NA,
                    modules = NA,
                    time = 24, ## in hours
                    mem = 4, ## in GB
                    cpus = 1,
                    gpu = FALSE,
                    partition = FALSE,
                    other_args = NULL
) {
  system = match.arg(system)
  
  # read the grid
  grid = read.delim(grid_file)
  
  # set up the script
  if (system == 'sockeye') {
    log_dir = file.path(dirname(base_dir), 'logs', basename(base_dir))
    header_lines = c(
      '#!/bin/bash',
      paste0('#PBS -l walltime=', time, ':00:00,select=1:n',
             ifelse(gpu, 'gpus=', 'cpus='), cpus,
             ':mem=', mem, 'gb'),
      paste0('#PBS -N ', job_name),
      paste0('#PBS -o ', log_dir, '/', job_name, '-^array_index^.out'),
      paste0('#PBS -e ', log_dir, '/', job_name, '-^array_index^.out'),
      ''
    )
    
    user = system("echo $USER", intern = TRUE)
    env = ifelse(is.na(env), 'invalid-smiles/env', env)
    env_lines = c(
      '# >>> conda initialize >>>',
      '# !! Contents within this block are managed by \'conda init\' !!',
      '__conda_setup="$(\'/project/st-ljfoster-1/miniconda3-py310/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)"', 
      'if [ $? -eq 0 ]; then',
      'eval "$__conda_setup"',
      'else',
      'if [ -f "/project/st-ljfoster-1/miniconda3-py310/etc/profile.d/conda.sh" ]; then',
      '. "/project/st-ljfoster-1/miniconda3-py310/etc/profile.d/conda.sh"',
      'else',
      'export PATH="/project/st-ljfoster-1/miniconda3-py310/bin:$PATH"',
      'fi',
      'fi',
      'unset __conda_setup',
      '# <<< conda initialize <<<',
      '',
      ifelse(grepl('^\\/', env),
             paste0('conda activate ', env),
             paste0('conda activate /scratch/st-ljfoster-1/', env)
      ),
      ''
    )
  } else if (system == 'argo') {
    log_dir = file.path(dirname(base_dir), 'logs', basename(base_dir))
    header_lines = c(
      '#!/bin/bash',
      paste0('#SBATCH --time=', time, ':00:00'),
      paste0('#SBATCH --job-name=', job_name),
      paste0('#SBATCH --output=', log_dir, '/%x-%j-%a.out'),
      paste0('#SBATCH --mem=', mem, 'G'),
      paste0('#SBATCH --cpus-per-task=', cpus),
      ifelse(gpu, paste0('#SBATCH --gres=gpu:1'), ''),
      ''
    )
    
    env_lines = c(
      '# >>> conda initialize >>>',
      '# !! Contents within this block are managed by \'conda init\' !!',
      '__conda_setup="$(\'/Genomics/skinniderlab/conda/miniconda3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)"',
      'if [ $? -eq 0 ]; then',
      'eval "$__conda_setup"',
      'else',
      'if [ -f "/Genomics/skinniderlab/conda/miniconda3/etc/profile.d/conda.sh" ]; then',
      '. "/Genomics/skinniderlab/conda/miniconda3/etc/profile.d/conda.sh"',
      'else',
      'export PATH="/Genomics/skinniderlab/conda/miniconda3/bin:$PATH"',
      'fi',
      'fi',
      'unset __conda_setup',
      '# <<< conda initialize <<<',
      '',
      ifelse(grepl('^\\/', env),
             paste0('conda activate ', env),
             paste0('conda activate ', file.path(base_dir, env))
      ),
      '',
      'JOB_SIZE=$1',
      ''
    )
  } else {
    stop('not sure how to write a sh file for: ', system)
  }
  
  # set up the final part of the script, which is platform-agnostic
  idx_var = switch(system,
                   'argo' = 'SLURM_ARRAY_TASK_ID',
                   'sockeye' = 'PBS_ARRAY_INDEX')
  run_lines = c(
    'cd ~/git/invalid-smiles-analysis',
    '',
    paste0('START=$((($', idx_var, '-1)*$JOB_SIZE + 1))'),
    paste0('STOP=$((($', idx_var, '-1)*$JOB_SIZE+$JOB_SIZE))'),
    'for i in $( seq $START $STOP ); do',
    paste0('GRID_FILE=', grid_file),
    paste0('LINE_IDX=$((i + 1))'),
    'LINE=`sed "${LINE_IDX}q;d" $GRID_FILE`',
    'IFS=$\'\\t\' PARAMS=($LINE)',
    map_chr(seq_len(ncol(grid)), ~ {
      col_idx = .x
      colname = colnames(grid)[col_idx]
      param_line = paste0(colname, '=${PARAMS[', col_idx - 1, ']}')
      param_line
    }),
    '',
    switch(gsub("^.*\\.", "", trimws(inner_file)),
           'R' = paste0('Rscript ', inner_file, ' \\'),
           'py' = paste0('python ', inner_file, ' \\')
    ),
    map_chr(seq_len(ncol(grid)), ~ {
      col_idx = .x
      colname = colnames(grid)[col_idx]
      if (col_idx < ncol(grid)) {
        arg_line = paste0('  --', colname, ' $', colname, ' \\')
      } else {
        arg_line = paste0('  --', colname, ' $', colname)
      }
      arg_line
    }),
    'done'
  )
  
  # write to file
  lines = c(header_lines,
            env_lines,
            run_lines)
  sh_file = switch(system,
                   argo = sh_file,
                   sockeye = gsub("\\.sh", "", sh_file) %>%
                     paste0(., '.torque.sh'))
  sh_dir = dirname(sh_file)
  if (!dir.exists(sh_dir))
    dir.create(sh_dir, recursive = TRUE)
  writeLines(lines, sh_file)
}
