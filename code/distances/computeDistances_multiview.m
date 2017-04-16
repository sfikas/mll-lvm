function D = computeDistances_multiview(Yin_pre, scale)
% Compute a single distance matrix, fusionning distances
% for all views.
%
% An optional scale parameter is used to scale the contribution of certain
% views. The distance computed for _scalar_ views (ie clinical scores in
% the medical imaging context) is multiplied by 'scale'.
%
% G.Sfikas June 2016
if(~exist('scale', 'var'))
    scale = 1;
end
count = 0;
Yin = cell(1, 1);
for i = 1:numel(Yin_pre)
    if(~isempty(Yin_pre{i}))
        count = count + 1;
        Yin{count} = Yin_pre{i};
    end
end
V = numel(Yin);
for i = 1:V
    Dp{i} = distmat(Yin{i},size(Yin{i}, 2),size(Yin{i}, 1));
end
% normalize each distance matrix
for i = 1:V
    Dp{i} = Dp{i} - mean(Dp{i}(:));
    Dp{i} = Dp{i} / std(Dp{i}(:));
end
% merge to a single D
w = ones(1, V);
if(exist('scale', 'var') && ~isempty(scale))
    for i = 1:V
        if(size(Yin{i}, 1) == 1) %clinical scores
            w(i) = scale; %unnecessary?
        end
    end
end
D = zeros(size(Dp{1}));
for i = 1:V
    D = D + w(i)*Dp{i};
end
D = D + abs(min(D(:)));
return;