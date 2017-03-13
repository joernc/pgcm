include("kernel.jl")

# path to folder containing param.h5 and <time step>.h5
path = ARGS[1]

# time step to start from
i0 = parse(Int, ARGS[2])

# load model setup
m = load(path)

# load model state
s = load(path, i0)

# time stepping
while s.i < 2*m.i1 #!!!
  # time step
  timestep!(m, s)
  # save every 1000 time steps
  if s.i%1000 == 0
    save(s, path)
    println("saved")
  end
end
