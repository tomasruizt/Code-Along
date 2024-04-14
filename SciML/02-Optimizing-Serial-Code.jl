A = rand(100,100)
B = rand(100,100)
C = rand(100,100)
using BenchmarkTools
function inner_rows!(C,A,B)
  for j in 1:100, i in 1:100
    val = A[i,j] + B[i,j]
    C[i,j] = val[1]
  end
end
@btime inner_rows!(C,A,B)