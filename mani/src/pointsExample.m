x = 5*randn(2000,1)+2;
y = 2*randn(2000,1)+3;
z = randn(2000,1)+5;

point = [x,y,z];

figure,scatter3(x,y,z)

[ T1 ] = mapperTopo_show( point );

figure,imshow(T1)


figure,imshow([T1,zeros(2000,2000);zeros(2000,2000),T])
