# see https://www.scipopt.org/doc-6.0.2/html/PARAMETERS.php for more information

solver_options = {
  # branching score function ('s'um, 'p'roduct, 'q'uotient)
  # [type: char, advanced: TRUE, range: {spq}, default: p]
  'branching/scorefunc' : 'p',
  # branching score factor to weigh downward and upward gain prediction in sum score function
  # [type: real, advanced: TRUE, range: [0,1], default: 0.167]
  'branching/scorefac' : 0.167,
  # maximal time in seconds to run
  # [type: real, advanced: FALSE, range: [0,1e+20], default: 1e+20]
  #'limits/time' : 1e+20
  }

def get_solver_options(s_options_dict: dict)->str:
  s_options = ""
  for k,v in s_options_dict.items():
    s_options += f"{k} = {v}\n"
  return s_options
  
