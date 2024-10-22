using YAML

# Check if a filename is provided as a command line argument
if length(ARGS) < 1
    println("Usage: julia SBGv001_L2T_STARS.jl <runconfig.yaml>")
    exit(1)  # Exit with an error code
end

# Get the filename from the first command line argument
runconfig_filename = ARGS[1]

# Load the YAML file
runconfig_dict = YAML.load_file(runconfig_filename)

# Print the contents in YAML format
println(YAML.write(runconfig_dict))
