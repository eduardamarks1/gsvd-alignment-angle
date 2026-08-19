using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra
using Random
using Statistics
using MLDatasets
using Plots

function wire_size(A, B)
    m, _ = size(A)
    p, _ = size(B)
    _, _, _, D1, D2, _ = svd(A, B)
    _, k_plus_l = size(D1)

    k = 0
    for i in 1:min(m, k_plus_l)
        D1[i, i] == 1 ? k += 1 : break
    end

    l = k_plus_l - k
    r = 0
    for i in 1:l
        D2[i, k + i] != 1 ? r += 1 : nothing
    end

    bl = k
    br = l - r
    wl = m - k - r
    wr = p - l

    return bl, br, wl, wr, r
end

function permutation(wr, r, br, D2_julia)
    P = [zeros(r, wr)  I(r)         zeros(r, br);
         zeros(br, wr) zeros(br, r) I(br);
         I(wr)         zeros(wr, r) zeros(wr, br)]

    P_til = [zeros(wr, r) zeros(wr, br) I(wr);
             I(r)        zeros(r, br)  zeros(r, wr);
             zeros(br, r) I(br)        zeros(br, wr)]

    return P, P_til * D2_julia
end

function Our_SVD(A, B)
    bl, br, wl, wr, r = wire_size(A, B)
    U, V, Q, D1, D2_julia, R0 = svd(A, B)

    P, D2_our = permutation(wr, r, br, D2_julia)
    V_til = V * P

    D1_til = D1[bl + 1:bl + r, bl + 1:bl + r]
    D2_til = D2_our[wr + 1:wr + r, bl + 1:bl + r]
    H = R0 * Q'

    return U, V_til, H, D1_til, D2_til, D1, D2_our, bl, br, wl, wr, r
end

function get_nonzero_per_column(M)
    return [begin
        col = M[:, j]
        idx = findfirst(!iszero, col)
        idx === nothing ? zero(eltype(M)) : col[idx]
    end for j in 1:size(M, 2)]
end

function sort_and_rebuild(U, V, D1, D2, D1_til, D2_til, H, bl, br, wl, wr, r)
    d1_vals = get_nonzero_per_column(D1_til)
    d2_vals = get_nonzero_per_column(D2_til)
    p = sortperm(d2_vals)

    d1_sorted = d1_vals[p]
    d2_sorted = d2_vals[p]

    H_sorted = copy(H)
    H_sorted[bl + 1:bl + r, :] = H[bl + 1:bl + r, :][p, :]

    U_sorted = copy(U)
    U_sorted[:, bl + 1:bl + r] = U[:, bl + 1:bl + r][:, p]

    V_sorted = copy(V)
    V_sorted[:, wr + 1:wr + r] = V[:, wr + 1:wr + r][:, p]

    D1_til_sorted = Diagonal(d1_sorted)
    D2_til_sorted = Diagonal(d2_sorted)

    D1_sorted = [I(bl)          zeros(bl, r)    zeros(bl, br);
                 zeros(r, bl)   D1_til_sorted   zeros(r, br);
                 zeros(wl, bl)  zeros(wl, r)    zeros(wl, br)]

    D2_sorted = [zeros(wr, bl)  zeros(wr, r)    zeros(wr, br);
                 zeros(r, bl)   D2_til_sorted   zeros(r, br);
                 zeros(br, bl)  zeros(br, r)    I(br)]

    return U_sorted, V_sorted, D1_til_sorted, D2_til_sorted, D1_sorted, D2_sorted, H_sorted
end

function infer_block_sizes(D1, D2)
    nrows1, ncols1 = size(D1)
    nrows2, ncols2 = size(D2)
    @assert ncols1 == ncols2

    bl = 0
    for i in 1:min(nrows1, ncols1)
        D1[i, i] == 1 ? bl += 1 : break
    end

    br = 0
    for j in 0:min(nrows2, ncols2) - 1
        D2[nrows2 - j, ncols2 - j] == 1 ? br += 1 : break
    end

    r = ncols1 - bl - br
    wl = nrows1 - bl - r
    wr = nrows2 - br - r
    return bl, br, wl, wr, r
end

function to_intersection(D1, D2)
    bl, br, wl, wr, r = infer_block_sizes(D1, D2)
    D1_int = copy(D1)
    D2_int = copy(D2)

    if bl > 0
        D1_int[1:bl, 1:bl] .= 0
    end
    if br > 0
        D2_int[wr + r + 1:wr + r + br, bl + r + 1:bl + r + br] .= 0
    end

    return D1_int, D2_int
end

function sample_digit_pair(digitA, digitB, nA, nB; split=:train, seed=1234)
    imgs, labels = MNIST(split=split)[:]
    imgs = Float64.(imgs) ./ 255

    idxA = findall(labels .== digitA)
    idxB = findall(labels .== digitB)
    Random.seed!(seed)
    idxA = Random.shuffle(idxA)[1:nA]
    idxB = Random.shuffle(idxB)[1:nB]

    A = hcat([vec(imgs[:, :, i]) for i in idxA]...)
    B = hcat([vec(imgs[:, :, i]) for i in idxB]...)
    return A, B, idxA, idxB
end

function prepare_data(digitA, digitB; n=900, m=900, split=:train, seed=1234)
    A, B, _, _ = sample_digit_pair(digitA, digitB, n, m; split=split, seed=seed)
    mean_A = vec(mean(A, dims=2))
    mean_B = vec(mean(B, dims=2))
    A_centered = A .- mean_A
    B_centered = B .- mean_B

    U, V, H, D1_til, D2_til, D1, D2, bl, br, wl, wr, r =
        Our_SVD(A_centered', B_centered')
    U, V, _, _, D1, D2, H =
        sort_and_rebuild(U, V, D1, D2, D1_til, D2_til, H, bl, br, wl, wr, r)

    return A_centered, B_centered, mean_A, mean_B, U, V, D1, D2, H
end

function gsvd_coefficients(v, U, V, D1, D2, H; coordinate_space=:sample)
    D1_int, D2_int = to_intersection(D1, D2)
    c = vec(H' \ v)

    shared = findall(j -> any(!iszero, D1_int[:, j]) && any(!iszero, D2_int[:, j]), 1:size(D1_int, 2))
    c_shared = c[shared]

    alpha_A = (D1_int[:, shared]') \ c_shared
    alpha_B = (D2_int[:, shared]') \ c_shared

    coeff_A = coordinate_space == :sample ? U * alpha_A : alpha_A
    coeff_B = coordinate_space == :sample ? V * alpha_B : alpha_B

    theta_deg = atan(norm(coeff_B), norm(coeff_A)) * 180 / pi
    return coeff_A, coeff_B, theta_deg
end

function one_digit_from_test(digit; seed=2026)
    imgs, labels = MNIST(split=:test)[:]
    imgs = Float64.(imgs) ./ 255
    idx = findall(labels .== digit)
    Random.seed!(seed + digit)
    chosen = Random.shuffle(idx)[1]
    return vec(imgs[:, :, chosen]), chosen
end

function coeff_stats_line(coeff_A, coeff_B, digitA, digitB)
    nA = norm(coeff_A)
    nB = norm(coeff_B)
    side = nA < nB ? digitA : digitB
    return "norma via $digitA = $(round(nA, digits=3)), norma via $digitB = $(round(nB, digits=3)); predicao $side"
end

function plot_instance_panel(v, coeff_A, coeff_B, sample_digit, digitA, digitB)
    nA = norm(coeff_A)
    nB = norm(coeff_B)

    img = reshape(v, 28, 28)
    p_img = heatmap(
        img[:, end:-1:1]',
        axis=nothing,
        colorbar=false,
        framestyle=:none,
        aspect_ratio=1,
        c=:viridis,
        title="amostra do digito $sample_digit"
    )

    lim = maximum(abs, vcat(coeff_A, coeff_B))
    lim = lim == 0 ? 1.0 : lim

    p_x = plot(
        1:length(coeff_A),
        coeff_A;
        label="||x|| = $(round(nA, digits=2))",
        color=:steelblue,
        linewidth=1.5,
        xlabel="coordenada",
        ylabel="valor do coeficiente",
        title="x via digito $digitA",
        ylims=(-lim, lim),
        framestyle=:box,
        legend=:topright
    )
    hline!(p_x, [0.0]; color=:gray, linestyle=:dash, linewidth=1, label="")

    p_y = plot(
        1:length(coeff_B),
        coeff_B;
        label="||y|| = $(round(nB, digits=2))",
        color=:coral,
        linewidth=1.5,
        xlabel="coordenada",
        ylabel="valor do coeficiente",
        title="y via digito $digitB",
        ylims=(-lim, lim),
        framestyle=:box,
        legend=:topright
    )
    hline!(p_y, [0.0]; color=:gray, linestyle=:dash, linewidth=1, label="")

    return plot(p_img, p_x, p_y; layout=(1, 3), size=(1180, 330))
end
function run_coeff_histogram_experiment(; digitA=4, digitB=9, n=900, m=900,
                                         base_seed=1234, sample_seed=2026,
                                         coordinate_space=:sample,
                                         savepath="coeff_linhas_separadas_espaco_amostras_4_9.png")
    A, B, _, _, U, V, D1, D2, H = prepare_data(digitA, digitB; n=n, m=m, seed=base_seed)

    vA_raw, idxA = one_digit_from_test(digitA; seed=sample_seed)
    vB_raw, idxB = one_digit_from_test(digitB; seed=sample_seed)

    # Match the existing notebook convention: test images are raw [0, 1] vectors.
    coeff_A_for_A, coeff_B_for_A, theta_A =
        gsvd_coefficients(vA_raw, U, V, D1, D2, H; coordinate_space=coordinate_space)
    coeff_A_for_B, coeff_B_for_B, theta_B =
        gsvd_coefficients(vB_raw, U, V, D1, D2, H; coordinate_space=coordinate_space)

    pA = plot_instance_panel(vA_raw, coeff_A_for_A, coeff_B_for_A, digitA, digitA, digitB)
    pB = plot_instance_panel(vB_raw, coeff_A_for_B, coeff_B_for_B, digitB, digitA, digitB)
    fig = plot(
        pA,
        pB;
        layout=(2, 1),
        size=(1050, 720)
    )

    savefig(fig, savepath)

    println("Figura salva em: $savepath")
    println("digito $digitA, indice de teste $idxA: ", coeff_stats_line(coeff_A_for_A, coeff_B_for_A, digitA, digitB),
            ", angulo=$(round(theta_A, digits=2)) graus")
    println("digito $digitB, indice de teste $idxB: ", coeff_stats_line(coeff_A_for_B, coeff_B_for_B, digitA, digitB),
            ", angulo=$(round(theta_B, digits=2)) graus")

    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_coeff_histogram_experiment(sample_seed=rand(1:10^9))
end
