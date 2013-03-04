function rr = generic_rrate(D,ns,method)
    nimages = size(D,1);
    ii = 1:nimages;
    for q=1:nimages
        [sd,si] = sort(D(q,:),method);
        r(si) = ii;
        c = floor((q-1) / ns);
        rr(:, q) = r((c*ns+1):((c+1)*ns))';
    end
end
        
