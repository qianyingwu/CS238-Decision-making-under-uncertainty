### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ a901f6f4-44e3-4fe6-8e15-ecf13ff136f7
using LinearAlgebra

# ╔═╡ d8635d7e-3bd5-471f-8f2c-37c4167c6aee
using Graphs

# ╔═╡ 7faa0ef5-ffb0-4920-aa7f-72e8c0b2e2ff
using Printf

# ╔═╡ 7655f186-a58c-4798-9c14-b3b0518ae1e7
using DataFrames

# ╔═╡ 7e13808c-11c2-451a-9dae-036ffe8b420e
using CSV

# ╔═╡ 7893c69b-6a00-433c-bcfe-7cbe2296b820
using TikzGraphs

# ╔═╡ 3e662fb0-9087-40f7-a3b4-9ca2aeb3dc79
using SpecialFunctions

# ╔═╡ 1562a863-b67a-4198-90c8-c4b402ab95f7
using TikzPictures # this is required for saving


# ╔═╡ ab890ecd-840e-4d83-8982-2970de2eaead
using Random

# ╔═╡ e65a5e90-a2b1-44be-9834-9894f842b379
# using NotebookToLaTeX

# ╔═╡ 5ceed2e7-9554-4f66-853c-10e27c27c6ea
# using Makie, CairoMakie, Plots,

# ╔═╡ a0d67519-bff9-432f-8041-7ca4e4090578
"""
    write_gph(dag::DiGraph, idx2names, filename)
Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format. idx2names is the ordering of the node names that you use. Basically, a dictionary that can map the node index to the node name.

"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

# ╔═╡ e5053686-d81d-4dec-8be6-7e056fe098a5
struct Variable 
	name::Symbol 
	# name::Symbol = :fixed
	r::Int # number of possible values 
end

# ╔═╡ 675b7af2-a0a3-4986-8537-1aac92c5c4d5
function sub2ind(siz, x) 
	k = vcat(1, cumprod(siz[1:end-1])) 
	return dot(k, x .- 1) + 1 
end


# ╔═╡ c43ae5ea-a6f9-4722-9e87-ace7af33caff
function statistics(vars, G, D::Matrix{Int})
	n = size(D, 1) 
	r = [vars[i].r for i in 1:n] 
	q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n] 
	M = [zeros(q[i], r[i]) for i in 1:n] 
	for o in eachcol(D) 
		for i in 1:n 
			k = o[i] 
			parents = inneighbors(G,i) 
			j = 1 
			if !isempty(parents) 
				j = sub2ind(r[parents], o[parents]) 
			end 
			M[i][j,k] += 1.0
		end
	end 
	return M
end

# ╔═╡ a36f6a64-762d-45f1-b8a5-61bfa2c785b3
function prior(vars, G) 
	n = length(vars) 
	r = [vars[i].r for i in 1:n] 
	q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n] 
	return [ones(q[i], r[i]) for i in 1:n] 
end

# ╔═╡ 2184a5f4-b0cf-4c71-b15f-b359b5eab8f2
function bayesian_score_component(M, α) 
	p = sum(loggamma.(α + M)) 
	p -= sum(loggamma.(α)) 
	p += sum(loggamma.(sum(α,dims=2))) 
	p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2))) 
	return p 
end

# ╔═╡ affac3a1-64ef-4530-840b-e80d1a4a3cd6
function bayesian_score(vars, G, D) 
	n = length(vars) 
	M = statistics(vars, G, D) 
	α = prior(vars, G) 
	return sum(bayesian_score_component(M[i], α[i]) for i in 1:n) 
end

# ╔═╡ dc3a8f64-ce2a-450c-9136-b9aa35b77534
struct LocalDirectedGraphSearch 
	G # initial graph 
	k_max # number of iterations 
end

# ╔═╡ 9281c407-40e8-4337-a366-cbd5dccc6b2b
function rand_graph_neighbor(G) 
	n = nv(G) 
	i = rand(1:n) 
	j = mod1(i + rand(2:n)-1, n)
	G′ = copy(G)
	has_edge(G, i, j) ? rem_edge!(G′, i, j) : add_edge!(G′, i, j) 
	return G′ 
end

# ╔═╡ b25a5cec-af28-4d0b-a4c0-c5ca0b69f018
function fit(method::LocalDirectedGraphSearch, vars, D)
	G = method.G 
	y = bayesian_score(vars, G, D) 
	for k in 1:method.k_max
		G′ = rand_graph_neighbor(G)
		y′ = is_cyclic(G′) ? -Inf : bayesian_score(vars, G′, D)
		if y′ > y
			y, G = y′, G′
		end		
	end 
		return G
end

# ╔═╡ 1c9282b4-be1d-4fb4-a84f-93dd42e3740d
struct K2Search 
	ordering::Vector{Int} # variable ordering 
end

# ╔═╡ 2fe1fefa-f638-4f64-82a1-0a4ee1c6be92
function fit(method::K2Search, vars, D)
	G = SimpleDiGraph(length(vars)) 
	for (k,i) in enumerate(method.ordering[2:end])
		y = bayesian_score(vars, G, D)
		while true

			y_best, j_best = -Inf, 0
			for j in method.ordering[1:k] 
				if !has_edge(G, j, i) 
					add_edge!(G, j, i) 
					y′ = bayesian_score(vars, G, D) 
					if y′ > y_best 
						y_best, j_best = y′, j 
					end 
					rem_edge!(G, j, i) 
				end
			end 
			if y_best > y
				y = y_best
				add_edge!(G, j_best, i) 
			else
				break
			end
		end
	end
	return G
end

# ╔═╡ 477153f0-7309-4487-83c5-128fd364fd6b
struct EnhanceK2Search
	not_important::Int
end

# ╔═╡ df5b5749-2330-4ca3-9262-fd707cdef26c
function compuateBayesianScore(g, D, vars)
	nVals = size(D, 1)
	nSamples = size(D, 2)

	# Compute the number of values for each variable
	nValCnts = [vars[i].r for i in 1:nVals]

	# Compute the number of instantiations of parents for each variable 
	nParents = [1 for i in 1:nVals]
	for i in 1:nVals
		base = 1
		for j in inneighbors(g,i)
			base *= nValCnts[j]
		end
		nParents[i] = base
	end

	# Compute α where α is the pseodocounts
	# Assume uniform distrbution for the prior 
	α = [[[1 for k in 1:nValCnts[i]] for j in 1:nParents[i]] for i in 1:nVals]

	# Compute M where M the actual counts
	m = [[[0 for k in 1:nValCnts[i]] for j in 1:nParents[i]] for i in 1:nVals]
	for j in 1:nSamples
		for i in 1:nVals
			val = D[i,j]
			
			# parentInstiation = []
			# total = []
			index = 0
			base = 1
			for parent in inneighbors(g, i)
				# push!(parentInstiation, D[parent,j])
				# push!(total, nValCnts[parent])
				index += (D[parent,j]-1) * base
				base *= nValCnts[parent]
			end
			index += 1
			m[i][index][val] += 1
			# if length(total) == 0
			# 	m[i][1][val] += 1
			# else
			# 	index = 0
			# 	base = 1
			# 	for i in 1:length(total)
			# 		index +=  (parentInstiation[i] - 1) * base
			# 		base *= total[i]
			# 	end
			# 	index += 1
			# 	m[i][index][val] += 1
			# end
		end
	end

	score = 0.0
	for i in 1:nVals
		for j in 1:nParents[i]
			αTmp = 0
			mTmp = 0
			for k in 1:nValCnts[i]
				score += loggamma(α[i][j][k] + m[i][j][k])
				score -= loggamma(α[i][j][k])
				αTmp += α[i][j][k]
				mTmp += m[i][j][k]
			end
			score += loggamma(αTmp)
			score -= loggamma(αTmp + mTmp)
		end
	end
	return score
			
end

# ╔═╡ c7076e99-c316-4748-9daf-8e6c9e7bd0d6
function incrementalScore(D, from, to, vars, g)
	nSamples = size(D, 2)
	nVals = size(D, 1)
	# Compute the number of values for each variable
	nValCnts = [vars[i].r for i in 1:nVals]

	# Compute the number of instantiations of parents for each variable 
	nParents = [1 for i in 1:nVals]
	for i in 1:nVals
		base = 1
		for j in inneighbors(g,i)
			base *= nValCnts[j]
		end
		nParents[i] = base
	end
	
	# calulate m[to, diffInstations, k]
	newM = [[0 for k in 1:nValCnts[to]] for j in 1:nParents[to]]
	oldM = [[0 for k in 1:nValCnts[to]] for j in 1:nParents[to]/nValCnts[from]]

	# compute oldM
	rem_edge!(g, from, to)
	for j in 1:nSamples
		val = D[to,j]
		index = 0
		base = 1
		for parent in inneighbors(g, to)
			index += (D[parent,j]-1) * base
			base *= nValCnts[parent]
		end
		index += 1
		oldM[index][val] += 1
	end

	# compute newM
	add_edge!(g, from, to)
	for j in 1:nSamples
		val = D[to,j]
		index = 0
		base = 1
		for parent in inneighbors(g, to)
			index += (D[parent,j]-1) * base
			base *= nValCnts[parent]
		end
		index += 1
		newM[index][val] += 1
	end


	# Compute diff
	diffInstations = nParents[to] - nParents[to] / nValCnts[from]
	diff = 0
	diff += diffInstations * (loggamma(nValCnts[to])) - diffInstations * nValCnts[to] * (loggamma(nValCnts[to])) 

	# println(nParents[to], ' ', nValCnts[from])
	for j in 1:Int(nParents[to]/nValCnts[from])
		mTmp = 0
		for k in 1:nValCnts[to]
			mTmp += oldM[j][k]
			diff -= loggamma(1 + oldM[j][k])
		end
		diff -= -loggamma(nValCnts[to] + mTmp)
	end

	for j in 1:nParents[to]
		mTmp = 0
		for k in 1:nValCnts[to]
			mTmp += newM[j][k]
			diff += loggamma(1 + newM[j][k])
		end
		diff += -loggamma(nValCnts[to] + mTmp)
	end
	
	return diff
end

# ╔═╡ d8cd8c7e-4cdd-471f-b245-3c30dca782ce
function k2Search(D, indexes, vars)
	# nRow denotes the number of variables in the dataset.
	nRow = size(D, 1)
	# nCol denotes the number of samples in the dataset.
	nCol = size(D, 2)
	
	# run the loops and get the best graph in a greedy way
	g = SimpleDiGraph(nRow)
	y = compuateBayesianScore(g, D, vars)
	for i in 2:nRow
		while true
			yBest = -Inf
			jBest = -1
			for j in 1:(i-1)
				if !has_edge(g, indexes[j], indexes[i])
					add_edge!(g, indexes[j], indexes[i])
					# yTmp = compuateBayesianScore(g, D, vars)
					 yTmp = y + incrementalScore(D,indexes[j], indexes[i],vars, g)
					if yTmp > yBest
						yBest = yTmp
						jBest = j
					end
					rem_edge!(g, indexes[j], indexes[i])
				end
			end
			if yBest > y
				y = yBest
				add_edge!(g, indexes[jBest], indexes[i])
			else
				break
			end
		end
	end
	return g
end
	

# ╔═╡ c8f92dfa-468d-40c7-bfd4-65710850c3e6
function fit(method::EnhanceK2Search, vars, D)
	# Use 70% of D for learning, the other 20% for score comparison between different viariable ordering
	ncol = size(D,2)
	cols = collect(1:ncol)
	colsL = Random.randsubseq(cols, 0.6)
	colsC = setdiff(cols, colsL)
	DL = D[:, colsL]
	DC = D[:, colsC]

	# random permute a variable ordering
	nvars = size(D,1)
	varidx = collect(1:nvars)

	score_best = -Inf
	G_best = SimpleDiGraph(nvars)
	npermute = 1000
	
	for i in 1:npermute
		indexes = shuffle(varidx)
		G = k2Search(D, indexes, vars)
		# generate a BN using K2 and the subset of data DL
		# G = fit(method, vars, DL)
	
		# compute the score using the subset of data DC
		# wait, using the complete data set
		score = compuateBayesianScore(G, D, vars)
	
		if score_best < score
			score_best, G_best = score, G
		end
	end

	return G_best

end

# ╔═╡ 53df6b43-b6c2-4352-b574-848273c1b3cb
function preprocess(D)
	nRow = size(D, 1)
	nCol = size(D, 2)
	for i in 1:nRow
		uniques = unique(D[i,:])
		map = Dict()
		for j in 1: length(uniques)
			map[uniques[j]] = j
		end
		for j in 1: nCol
			D[i,j] = map[D[i,j]]
		end
	end
	return D
end
		
	

# ╔═╡ 27f2e68f-afb7-4644-a47c-30617eeb79f5
function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
	df = DataFrame(CSV.File(infile))
	
	nrow = size(df, 1)
	ncol = size(df, 2)
	vars = Vector{Variable}()
	for i in 1:ncol
		nunique = length(unique(df[:,i]))
		name = names(df)[i]
		push!(vars, Variable(Symbol(name), nunique))
	end
	# print(vars)
	
	# return vars
	# return first(df, 6)
	idx = collect(1:ncol)
	idx2names = Dict(idx .=> names(df))
	g = SimpleDiGraph(ncol) 

	k_max = 300
	# method = LocalDirectedGraphSearch(G, k_max)

	ordering = collect(1:ncol)
	# ordering = [1, 2, 3, 3, 2, 1]
	# method = K2Search(ordering)
	# G = fit(method, vars, D)

	method = EnhanceK2Search(0)
	
	D = (Matrix(df))
	D = permutedims(D)
	D = preprocess(D)
	
	# α = prior(vars, G) 
	@time g = fit(method, vars, D)


	# g = SimpleDiGraph(6)
	# add_edge!(g, 1, 2)
	# add_edge!(g, 3, 4)
	# add_edge!(g, 5, 6)

	# g = k2Search(D)
	
	bs2 = compuateBayesianScore(g, D, vars)
	println(bs2)
	#bs = 0

	# CSV.write("test_output.csv", df)
	write_gph(g::DiGraph, idx2names, outfile)

	t = TikzGraphs.plot(g, names(df))
	prefixIndex = 0
	posfixIndex = 0
	for i in 1:length(infile)
		if infile[i] == '/'
			prefixIndex = i
		end
		if infile[i] == '.'
			posfixIndex = i
		end
	end
 	suffix = chop(infile, head = prefixIndex, tail=length(infile)- posfixIndex+1)
	filename = "graph_" * String(suffix)
	TikzPictures.save(PDF(filename), t)
	TikzPictures.save(SVG(filename), t)
	TikzPictures.save(TEX(filename), t)
	return vars
end

# ╔═╡ c927f9dc-32e9-4f5b-ab0e-d640ec6cf753
for i = 1:3
	if i == 1
		ARGS = ["data/large.csv", "data/large.gph"]
	elseif i == 2
		ARGS = ["data/medium.csv", "data/medium.gph"]
	else
		ARGS = ["data/small.csv", "data/small.gph"]
	end
	
	if length(ARGS) != 2
	    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
	end
	
	inputfilename = ARGS[1]
	outputfilename = ARGS[2]
	compute(inputfilename, outputfilename)
end

# ╔═╡ e7bd2072-f8ce-43ac-b65f-e53f206c7901
# notebooktolatex("notebook.jl", template=:book)

# ╔═╡ d9707d26-16b1-4b2a-8245-9999548044cb
# t = TikzGraphs.plot(G, ["A", "B", "C", "D"])
# TikzPictures.save(PDF("graph"), t)
# TikzPictures.save(SVG("graph"), t)
# TikzPictures.save(TEX("graph"), t)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
TikzGraphs = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"

[compat]
CSV = "~0.10.9"
DataFrames = "~1.4.4"
Graphs = "~1.7.4"
SpecialFunctions = "~2.1.7"
TikzGraphs = "~1.4.0"
TikzPictures = "~3.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "a2b9174522ac3ba4d53d9021feeaae2565c07c57"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "844b061c104c408b24537482469400af6075aae4"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "45b288af6956e67e621c5cbb2d75a261ab58300b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.20"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "8175fc2b118a3755113c8e68084dc1a9e63c61ee"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Poppler_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "e11443687ac151ac6ef6699eb75f964bed8e1faa"
uuid = "9c32591e-4766-534b-9725-b71a8799265b"
version = "0.87.0+2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "c02bd3c9c3fc8463d3591a62a378f90d2d8ab0f3"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.17"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6954a456979f23d05085727adb17c4551c19ecd1"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.12"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Tectonic]]
deps = ["Pkg"]
git-tree-sha1 = "0b3881685ddb3ab066159b2ce294dc54fcf3b9ee"
uuid = "9ac5f52a-99c6-489f-af81-462ef484790f"
version = "0.8.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TikzGraphs]]
deps = ["Graphs", "LaTeXStrings", "TikzPictures"]
git-tree-sha1 = "e8f41ed9a2cabf6699d9906c195bab1f773d4ca7"
uuid = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
version = "1.4.0"

[[deps.TikzPictures]]
deps = ["LaTeXStrings", "Poppler_jll", "Requires", "Tectonic"]
git-tree-sha1 = "4e75374d207fefb21105074100034236fceed7cb"
uuid = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
version = "3.4.2"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═a901f6f4-44e3-4fe6-8e15-ecf13ff136f7
# ╠═d8635d7e-3bd5-471f-8f2c-37c4167c6aee
# ╠═7faa0ef5-ffb0-4920-aa7f-72e8c0b2e2ff
# ╠═7655f186-a58c-4798-9c14-b3b0518ae1e7
# ╠═7e13808c-11c2-451a-9dae-036ffe8b420e
# ╠═7893c69b-6a00-433c-bcfe-7cbe2296b820
# ╠═3e662fb0-9087-40f7-a3b4-9ca2aeb3dc79
# ╠═1562a863-b67a-4198-90c8-c4b402ab95f7
# ╠═ab890ecd-840e-4d83-8982-2970de2eaead
# ╠═e65a5e90-a2b1-44be-9834-9894f842b379
# ╠═5ceed2e7-9554-4f66-853c-10e27c27c6ea
# ╠═a0d67519-bff9-432f-8041-7ca4e4090578
# ╠═e5053686-d81d-4dec-8be6-7e056fe098a5
# ╠═675b7af2-a0a3-4986-8537-1aac92c5c4d5
# ╠═c43ae5ea-a6f9-4722-9e87-ace7af33caff
# ╠═a36f6a64-762d-45f1-b8a5-61bfa2c785b3
# ╠═2184a5f4-b0cf-4c71-b15f-b359b5eab8f2
# ╠═affac3a1-64ef-4530-840b-e80d1a4a3cd6
# ╠═dc3a8f64-ce2a-450c-9136-b9aa35b77534
# ╠═9281c407-40e8-4337-a366-cbd5dccc6b2b
# ╠═b25a5cec-af28-4d0b-a4c0-c5ca0b69f018
# ╠═1c9282b4-be1d-4fb4-a84f-93dd42e3740d
# ╠═2fe1fefa-f638-4f64-82a1-0a4ee1c6be92
# ╠═477153f0-7309-4487-83c5-128fd364fd6b
# ╠═df5b5749-2330-4ca3-9262-fd707cdef26c
# ╠═c7076e99-c316-4748-9daf-8e6c9e7bd0d6
# ╠═d8cd8c7e-4cdd-471f-b245-3c30dca782ce
# ╠═c8f92dfa-468d-40c7-bfd4-65710850c3e6
# ╠═53df6b43-b6c2-4352-b574-848273c1b3cb
# ╠═27f2e68f-afb7-4644-a47c-30617eeb79f5
# ╠═c927f9dc-32e9-4f5b-ab0e-d640ec6cf753
# ╠═e7bd2072-f8ce-43ac-b65f-e53f206c7901
# ╠═d9707d26-16b1-4b2a-8245-9999548044cb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
