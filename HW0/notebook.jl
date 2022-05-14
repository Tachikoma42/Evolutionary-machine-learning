### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 6c77a4de-7a6b-4798-9918-4ef8f4b6b370
using Random

# ╔═╡ f18b78cf-4a4c-4bb4-a8ad-a0ac2fff8e28
md"This program randomly generate 5 strings then sort them using bubbleSort."

# ╔═╡ 202de2ab-6e66-43d8-a9cc-040b33e42818
function bubbleSort(tobeSorted)
    for i in 1:length(tobeSorted)-1
        for j in 1:length(tobeSorted)-1
            if cmp(tobeSorted[j],tobeSorted[j+1]) == 1
                # equal to 1 means that the first element is bigger than the second element
                # so we need to swap them
                # 0 , -1 means that the first element is smaller or equal to the second element
                # no need to swap
                tobeSorted[j],tobeSorted[j+1] = tobeSorted[j+1],tobeSorted[j]
            end
                
        end
    end
    return tobeSorted
end

# ╔═╡ 0525358c-e340-436c-9a53-852dc2c04fb1
md"the sort order in julia is number,uppercase then lowercase."

# ╔═╡ 4b1f13b6-bf9b-48d6-893c-00588c4d15d6
begin
	tobeSorted = String[]
		
		for i in 1:5 
		    push!(tobeSorted, randstring(6))
		end
	#@info bubbleSort(tobeSorted)
		bubbleSort(tobeSorted)
end

# ╔═╡ 0b595358-03bd-42a8-9810-abff8b26e520
md"Pluto.ji doesn't support print, so I have to use @info or just leave it there to display the result."

# ╔═╡ 1167d324-8f07-4958-bd84-54ab0f1dc728
md"The following is a test program using build-in mergeSort to check the result, the build-in sort function is from https://docs.julialang.org/en/v1/base/sort/"

# ╔═╡ 12baee62-e89b-444a-a5e3-9c8fb4dad4d9
# begin
# 	tobeSorted = String[]
	
# 	for i in 1:5 
# 	    push!(tobeSorted, randstring(6))
# 	end
	
# 	if bubbleSort(tobeSorted) == sort(tobeSorted, alg = MergeSort)
# 		@info "Sorting is correct"
	
# 	else
# 	    @info "Sorting is incorrect" tobeSorted
		
	 	
# 	end

# end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
"""

# ╔═╡ Cell order:
# ╠═f18b78cf-4a4c-4bb4-a8ad-a0ac2fff8e28
# ╠═6c77a4de-7a6b-4798-9918-4ef8f4b6b370
# ╠═202de2ab-6e66-43d8-a9cc-040b33e42818
# ╠═0525358c-e340-436c-9a53-852dc2c04fb1
# ╠═4b1f13b6-bf9b-48d6-893c-00588c4d15d6
# ╠═0b595358-03bd-42a8-9810-abff8b26e520
# ╠═1167d324-8f07-4958-bd84-54ab0f1dc728
# ╠═12baee62-e89b-444a-a5e3-9c8fb4dad4d9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
