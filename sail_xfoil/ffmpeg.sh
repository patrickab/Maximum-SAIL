#!/bin/bash

# Function to process files in a directory
recursive_ffmpeg() {
  local dir="./imgs"
  local benchmark_dir
  local output_file
  local run_index
  local run_index_string
  local extension
  local filename
  local new_filename
  # add necessary variables
  local benchmark_dir_name
  local benchmark_string
  local run_id
  local file

  if [ $# -eq 1 ]; then
    output_file="$1"  # Use the provided argument as the output filename
  fi
  
  # Process all files in the directory
  for benchmark_dir in "$dir"/*; do
    benchmark_dir_name=""
    # check if file is a directory - if it is a directory, enter & store name of the directory
    if [[ -d "$benchmark_dir" ]]; then
      benchmark_dir_name="${benchmark_dir##*/}"
      for run_index in "$benchmark_dir"/*; do
        run_index_string=""
        # check if file is a directory - if it is a directory, enter & store name of the directory
        if [[ -d "$run_index" ]]; then
          run_index_string=""
          # parse run_index including leading zeros
          run_index_string=$(printf "%03d" ${run_index##*/})
          run_index_string="${run_index##*/}"

          for file in "$run_index"/*; do
            benchmark_string=""
            filename_string=""
            if [[ -f "$file" ]]; then
              # Get only png files
              if [[ "${file##*.}" != "png" ]]; then
                continue
              fi
              # Get the filename without the extension
              filename_string="${file%.*}"
              # remove all directories above from filename string
              filename_string="${filename_string##*/}"
              # from the filename, extract benchmark_string by removing obj_ and the following number and the _
              # For example: obj_1_ obj_99_ or obj_999_ - only store the string, that follows after
              benchmark_string="${filename_string#obj_[0-9]*_}"


              echo ""
              echo "benchmark_dir_name: $benchmark_dir_name"
              echo "run_index_string: $run_index_string"
              echo "file: $file"
              echo "filename_string: $filename_string"
              echo "benchmark_string: $benchmark_string"
              echo "output_file: $output_file"
              echo ""


              # store all videos in the same folder by not adding the run_index_string

              # if recursive ffmpeg has been called with an argument
              if [ $# -eq 1 ]; then
                ffmpeg -framerate 5 -i "$dir/$benchmark_dir_name/$run_index_string/obj_%03d_$benchmark_string.png" -r 30 -pix_fmt yuv420p $dir/$benchmark_dir_name/$output_file
              fi
              cp "$(ls -v "$dir/$benchmark_dir_name/$run_index_string/obj_"*"$benchmark_string.png" | tail -n 1)" "$dir/$benchmark_dir_name/obj_$run_index_string"_"$benchmark_string"_final_state".png"
              rm $dir/$benchmark_dir_name/$run_index_string/*.png
              # break the current for loop, keep executing "for run_index in "$benchmark_dir"/*; do"
              break
            fi
          done
        fi
      done
    fi
  done
}

# Execute the recursive_ffmpeg function
if [ $# -eq 1 ]; then
  recursive_ffmpeg "$1"
else
  echo "No custom output filename provided. Using the default 'buffervid.mp4'."
  recursive_ffmpeg "lulu.mp4"
fi
