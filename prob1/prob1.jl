using Random, SparseArrays, Plots
include("arnoldi_GMRES.jl")

Random.seed!(0)
α_list = (1, 5, 10, 100)
n = 100

for α in α_list
  A, b = generate_A_b(α)
  (Q, H, z_✶, xₘ) = arnoldi_GRMS(A, b, 100, double_GS_MV)

  plot(1:100, norm(), yscale=:log10, label="||Ax - b||/||b||")
  savefig("gmres_alpha=$α")
end

function generate_A_b(α)
  A = sprand(n, n, 0.5)
  A = A + α*I
  A = A/norm(A, 1);
  b = randn(n);
  return (A, b)
end

