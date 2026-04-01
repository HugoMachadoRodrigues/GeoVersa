source("code/generate_wadoux_maps.R")

maps <- generate_wadoux_maps(
  context = wadoux_context,
  sample_size = 500,
  sampling = "simple_random",
  results_dir = "results/maps",
  krigingnet_params = krigingnet_params
)
