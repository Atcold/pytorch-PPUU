import re
import pandas as pd

pattern = re.compile(r'.*step (\d+) \| train: \[c: (\S+), l: (\S+), u: (\S+), a: (\S+), p: (\S+) \] \| test: \[c: (\S+), l:(\S+), u: (\S+), a: (\S+), p: (\S+)]')
df_header = (
    'Step',
    'TrPrx', 'TrLan', 'TrUnc', 'TrAct', 'TrLss',
    'TePrx', 'TeLan', 'TeUnc', 'TeAct', 'TeLss',
)
job_header = 'job name'

def load_log_file(file_name):
    
    with open(file_name, 'r') as policy_log_file:

        # Count nb of lines to skip
        last_header_line_nb = 0  # lines to be skipped
        for n, line in enumerate(policy_log_file):
            if job_header in line: last_header_line_nb = n
        
        # Read valid data
        policy_log_file.seek(0)  # go back to the top
        policy_log_list = list()
        for n, line in enumerate(policy_log_file):
            if n <= last_header_line_nb: continue
            match = re.match(pattern, line)
            policy_log_list.append(tuple(
                int(g) if i is 0 else float(g) for i, g in enumerate(match.groups())
            ))

    # Create data frame
    df = pd.DataFrame(data=policy_log_list, columns=df_header)

    return df
