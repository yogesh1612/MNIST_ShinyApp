#Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python3")
# This file configures the virtualenv and Python paths differently depending on
# the environment the app is running in (local vs remote server).

# # Edit this name if desired when starting a new app
# VIRTUALENV_NAME = 'example_env_name'
# 
# 
# # ------------------------- Settings (Do not edit) -------------------------- #
# 
# if (Sys.info()[['user']] == 'shiny'){
#   
#   # Running on shinyapps.io
#   Sys.setenv(PYTHON_PATH = 'python3')
#   Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME) # Installs into default shiny virtualenvs dir
#   Sys.setenv(RETICULATE_PYTHON = paste0('/home/shiny/.virtualenvs/', VIRTUALENV_NAME, '/bin/python'))
#   
# } else if (Sys.info()[['user']] == 'rstudio-connect'){
#   
#   # Running on remote server
#   Sys.setenv(PYTHON_PATH = '/opt/python/3.7.6/bin/python')
#   Sys.setenv(VIRTUALENV_NAME = paste0(VIRTUALENV_NAME, '/')) # include '/' => installs into rstudio-connect/apps/
#   Sys.setenv(RETICULATE_PYTHON = paste0(VIRTUALENV_NAME, '/bin/python'))
#   
# } else {
#   
#   # Running locally
#   options(shiny.port = 7450)
#   Sys.setenv(PYTHON_PATH = 'python3')
#   Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME) # exclude '/' => installs into ~/.virtualenvs/
#   # RETICULATE_PYTHON is not require
#   
#   
# }

# devtools::install_github("rstudio/rsconnect", ref='737cd48')
# 
# virtualenv_create(envname = "python_environment", python= "python3")
# virtualenv_install("python_environment", packages = c('keras','h5py','pandas','numpy','scipy','scikit-learn', 'tensorflow'))
# reticulate::use_virtualenv("python_environment", required = TRUE)