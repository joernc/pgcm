using Documenter

push!(LOAD_PATH, "../src")
include("../src/kernel.jl")

makedocs()
