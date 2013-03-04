function f = ggdinv(u,a,b)
    pos = find(u <= 0.5);
    neg = find(u >  0.5);
    f(pos) = -a*power(gaminv(1-2*u(pos),1/b,1),1/b);
    f(neg) = a*power(gaminv(2*u(neg)-1,1/b,1),1/b);
end

