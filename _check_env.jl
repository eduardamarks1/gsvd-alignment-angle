using Pkg
# all packages present anywhere in the depot (installed versions)
names = sort(unique([info.name for (uuid, info) in Pkg.dependencies()]))
want = ["MLDatasets","Plots","CSV","DataFrames","FileIO","IJulia"]
println("=== wanted packages present in depot ===")
for w in want
    println(w, " => ", w in names ? "INSTALLED" : "missing")
end
println("=== total installed packages: ", length(names), " ===")
