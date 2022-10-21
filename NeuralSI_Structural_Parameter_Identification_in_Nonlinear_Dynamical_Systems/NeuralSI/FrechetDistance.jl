

function frdist(p, q, xlist)
    # p = Array(p)
    # q = Array(q)
    p = cat(xlist,Array(p);dims=2)
    q = cat(xlist,Array(q);dims=2)
    len_p = length(p[:,1])
    len_q = length(q[:,1])

    if len_p == 0 || len_q == 0
        error("Input curves are empty.")
    end
    # if len_p != len_q || length(p[1]) != length(q[1])
    #     error("Input curves do not have the same dimensions.")
    # end
    ca = (ones(len_p, len_q) .* -1)
    dist = _c(ca, len_p, len_q, p, q)
    return dist
end

function _c(ca, i, j, p, q)
    if ca[i, j] > -1
        return ca[i, j]
    elseif i == 1 && j == 1
        ca[i, j] = norm(p[i,:]-q[j,:])
    elseif i > 1 && j == 1
        ca[i, j] = max(_c(ca, i-1, 1, p, q), norm(p[i,:]-q[j,:]))
    elseif i == 1 && j > 1
        ca[i, j] = max(_c(ca, 1, j-1, p, q), norm(p[i,:]-q[j,:]))
    else i > 1 && j > 1
        ca[i, j] = max(
            min(_c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q),),
            norm(p[i,:]-q[j,:]) )
    end

    return ca[i, j]
end
