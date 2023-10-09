library(readr)


# creates a list of dataframes
read_csv_files <- function(directory, init, n, prefix, domain) {
  data_frames <- list()
  
  for (i in init:n) {                                            # Loop through files from 1 to n
    file_name <- paste0(prefix, "_", domain, "_", i, ".csv")     # Generate the file name based on the provided prefix and index
    file_path <- file.path(directory, file_name)                 # Construct the full file path
    if (file.exists(file_path)) {
      data_frames[[file_name]] <- read.csv(file_path)            # Read the CSV file and store it in the list
      data_frames$file_name
    } else {
      cat("File not found:", file_path, "\n")
    }
  }
  
  # Return the list of data frames
  return(data_frames)
}

# Example usage:
directory <- "C:/Users/patri/Desktop/Thesis/archives_xfoil"
init <- 0
n <- 0

setwd(directory)


domain1 = "custom"
domain2 = "vanilla"
domain3 = "random"

obj <- read_csv_files(directory, init=init,n=n, prefix="obj_archive", domain=domain1)
pred_error <- read_csv_files(directory, init=init, n=n, prefix="pred_error_archive", domain=domain1)
verified_obj <- read_csv_files(directory, init=init, n=n, prefix= "verified_obj_archive", domain=domain1)
unverified_obj <- read_csv_files(directory, init=init, n=n, prefix="unverified_obj_archive", domain=domain1)

list2env(obj, envir = .GlobalEnv)
list2env(pred_error, envir = .GlobalEnv)
list2env(verified_obj, envir = .GlobalEnv)
list2env(unverified_obj, envir = .GlobalEnv)

combined_obj <- do.call(rbind, obj)
combined_pred_error <- do.call(rbind, pred_error)
combined_verified_obj <- do.call(rbind, verified_obj)
combined_unverified_obj <- do.call(rbind, unverified_obj)

custom_combined_obj <- combined_obj
custom_combined_verified_obj <- combined_verified_obj

write.csv(combined_obj, file = paste0("combined_obj", "_", domain=domain1, ".csv"), row.names = FALSE)
write.csv(combined_pred_error, file = paste0("combined_pred_error", "_", domain=domain1, ".csv"), row.names = FALSE)
write.csv(combined_verified_obj, file = paste0("combined_verified_obj", "_", domain=domain1, ".csv"), row.names = FALSE)
write.csv(combined_unverified_obj, file = paste0("combined_unverified_obj", "_", domain=domain1, ".csv"), row.names = FALSE)




obj <- read_csv_files(directory, init=init,n=n, prefix="obj_archive", domain=domain2)
pred_error <- read_csv_files(directory, init=init, n=n, prefix="pred_error_archive", domain=domain2)
verified_obj <- read_csv_files(directory, init=init, n=n, prefix= "verified_obj_archive", domain=domain2)
unverified_obj <- read_csv_files(directory, init=init, n=n, prefix="unverified_obj_archive", domain=domain2)

list2env(obj, envir = .GlobalEnv)
list2env(pred_error, envir = .GlobalEnv)
list2env(verified_obj, envir = .GlobalEnv)
list2env(unverified_obj, envir = .GlobalEnv)

combined_obj <- do.call(rbind, obj)
combined_pred_error <- do.call(rbind, pred_error)
combined_verified_obj <- do.call(rbind, verified_obj)
combined_unverified_obj <- do.call(rbind, unverified_obj)

vanilla_combined_obj <- combined_obj
vanilla_combined_verified_obj <- combined_verified_obj

write.csv(combined_obj, file = paste0("combined_obj", "_", domain=domain2, ".csv"), row.names = FALSE)
write.csv(combined_pred_error, file = paste0("combined_pred_error", "_", domain=domain2, ".csv"), row.names = FALSE)
write.csv(combined_verified_obj, file = paste0("combined_verified_obj", "_", domain=domain2, ".csv"), row.names = FALSE)
write.csv(combined_unverified_obj, file = paste0("combined_unverified_obj", "_", domain=domain2, ".csv"), row.names = FALSE)




obj <- read_csv_files(directory, init=init,n=n, prefix="obj_archive", domain=domain3)
pred_error <- read_csv_files(directory, init=init, n=n, prefix="pred_error_archive", domain=domain3)
verified_obj <- read_csv_files(directory, init=init, n=n, prefix= "verified_obj_archive", domain=domain3)
unverified_obj <- read_csv_files(directory, init=init, n=n, prefix="unverified_obj_archive", domain=domain3)

list2env(obj, envir = .GlobalEnv)
list2env(pred_error, envir = .GlobalEnv)
list2env(verified_obj, envir = .GlobalEnv)
list2env(unverified_obj, envir = .GlobalEnv)

combined_obj <- do.call(rbind, obj)
combined_pred_error <- do.call(rbind, pred_error)
combined_verified_obj <- do.call(rbind, verified_obj)
combined_unverified_obj <- do.call(rbind, unverified_obj)

random_combined_obj <- combined_obj
random_combined_verified_obj <- combined_verified_obj

write.csv(combined_obj, file = paste0("combined_obj", "_", domain=domain3, ".csv"), row.names = FALSE)
write.csv(combined_pred_error, file = paste0("combined_pred_error", "_", domain=domain3, ".csv"), row.names = FALSE)
write.csv(combined_verified_obj, file = paste0("combined_verified_obj", "_", domain=domain3, ".csv"), row.names = FALSE)
write.csv(combined_unverified_obj, file = paste0("combined_unverified_obj", "_", domain=domain3, ".csv"), row.names = FALSE)

