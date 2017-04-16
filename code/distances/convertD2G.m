function G = convertD2G(Dmat, k)
% Create a binary NxN neighbourhood graph out of a distance matrix
% of the same size, and given the number of nearest neighbours k.
%
% G Sfikas June 2016
n = size(Dmat, 1);
[V ] = sort(Dmat, 1);
G = (Dmat <= repmat(V(k+1, :), n, 1) ) - eye(n);
G = G + G' >= 1;
G = double(G);
return;