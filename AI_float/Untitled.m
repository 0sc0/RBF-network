[xx1, nx1] = mapminmax(x1, -1, 1);
[xx2, nx2] = mapminmax(x2, -1, 1);
[xx3, nx3] = mapminmax(x3, -1, 1);
[yy1, ny1] = mapminmax(y1, -1, 1);
[yy2, ny2] = mapminmax(y2, -1, 1);
[yy3, ny3] = mapminmax(y3, -1, 1);
[yy4, ny4] = mapminmax(y4, -1, 1);
[yy5, ny5] = mapminmax(y5, -1, 1);
[yy6, ny6] = mapminmax(y6, -1, 1);

net = newrbe([xx1; xx2; xx3], [yy1; yy2; yy3; yy4 ;yy5; yy6]);

iip1 = mapminmax('apply',ip1, nx1);
iip2 = mapminmax('apply',ip2, nx2);
iip3 = mapminmax('apply',ip3, nx3);

nresult = sim(net, [iip1; iip2; iip3]);

result1 = mapminmax('reverse', nresult(1,:) , ny1);
result2 = mapminmax('reverse', nresult(2,:) , ny2);
result3 = mapminmax('reverse', nresult(3,:) , ny3);
result4 = mapminmax('reverse', nresult(4,:) , ny4);
result5 = mapminmax('reverse', nresult(5,:) , ny5);
result6 = mapminmax('reverse', nresult(6,:) , ny6);


