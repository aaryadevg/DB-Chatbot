-- DDL
CREATE TABLE public.aggregations (
	id serial4 NOT NULL,
	shift_date date NULL,
	shift_name varchar(10) NULL,
	machine_name varchar(50) NULL,
	production int4 NOT NULL DEFAULT 0,
	rejection int4 NOT NULL DEFAULT 0,
	machine_runtime_minute float4 NOT NULL DEFAULT 0,
	machine_downtime_minute float4 NOT NULL DEFAULT 0,
	ideal_speed float4 NOT NULL DEFAULT 0,
	actual_speed float4 NOT NULL DEFAULT 0,
	availability float4 NOT NULL DEFAULT 0,
	performance float4 NOT NULL DEFAULT 0,
	quality float4 NOT NULL DEFAULT 0,
	oee float4 NOT NULL DEFAULT 0,
	CONSTRAINT aggregations_pk PRIMARY KEY (id)
);


-- Sample Data
INSERT INTO aggregations
	(shift_date,shift_name,machine_name,production,rejection,machine_runtime_minute,machine_downtime_minute,ideal_speed,actual_speed,availability,performance,quality,oee) 
VALUES
	 ('2024-06-02','I','Machine 1',8656,0,345.0,510.0,52.0,50.367954,68.0,48.0,100.0,32.0),
	 ('2024-06-02','I','Machine 2',8000,0,188.0,510.0,45.0,41.4432,37.0,95.0,100.0,35.0),
	 ('2024-06-02','I','Machine 3',18200,15,362.0,510.0,55.0,46.398834,71.0,91.0,100.0,64.0),
	 ('2024-06-02','I','Machine 4',8137,10,377.0,510.0,45.0,42.208706,74.0,48.0,100.0,35.0),
	 ('2024-06-02','I','Machine 5',8430,0,255.0,510.0,70.0,64.89502,50.0,47.0,100.0,23.0),
	 ('2024-06-02','II','Machine 1',10001,0,241.0,510.0,52.0,49.597355,47.0,80.0,100.0,37.0),
	 ('2024-06-02','II','Machine 2',15025,0,352.0,510.0,45.0,41.711266,69.0,95.0,100.0,65.0),
	 ('2024-06-02','II','Machine 3',16129,15,348.0,510.0,55.0,45.503994,68.0,84.0,100.0,57.0),
	 ('2024-06-02','II','Machine 4',16241,10,387.0,510.0,45.0,42.123215,76.0,93.0,100.0,70.0),
	 ('2024-06-02','II','Machine 5',32557,160,350.0,510.0,70.0,65.61263,69.0,133.0,100.0,91.0),
	 ('2024-06-02','III','Machine 1',10875,0,215.0,420.0,52.0,49.333546,51.0,97.0,100.0,49.0),
	 ('2024-06-02','III','Machine 2',11034,0,261.0,420.0,45.0,40.625324,62.0,94.0,100.0,58.0),
	 ('2024-06-02','III','Machine 3',1743,0,9.0,420.0,55.0,0,2.0,352.0,100.0,7.0),
	 ('2024-06-02','III','Machine 4',11533,0,295.0,420.0,45.0,42.24004,70.0,87.0,100.0,60.0),
	 ('2024-06-02','III','Machine 5',1923,80,300.0,420.0,70.0,63.23863,71.0,9.0,96.0,6.0),
	 ('2024-06-03','I','Machine 1',16982,0,332.0,510.0,52.0,50.43306,65.0,98.0,100.0,63.0),
	 ('2024-06-03','I','Machine 2',11400,9,274.0,510.0,45.0,40.611008,54.0,92.0,100.0,49.0),
	 ('2024-06-03','I','Machine 3',16000,20,322.0,510.0,55.0,43.934586,63.0,90.0,100.0,56.0),
	 ('2024-06-03','I','Machine 4',15163,10,339.0,510.0,45.0,42.108757,66.0,99.0,100.0,65.0),
	 ('2024-06-03','III','Machine 1',13836,0,275.0,420.0,52.0,50.40335,65.0,97.0,100.0,63.0),
	 ('2024-06-03','III','Machine 3',15481,0,309.0,420.0,55.0,46.879227,74.0,91.0,100.0,67.0),
	 ('2024-06-03','III','Machine 4',11800,0,298.0,420.0,45.0,42.068336,71.0,88.0,100.0,62.0),
	 ('2024-06-04','II','Machine 1',14377,124,281.0,510.0,52.0,50.25093,55.0,98.0,99.0,53.0),
	 ('2024-06-04','II','Machine 2',9849,0,276.0,510.0,45.0,42.19032,54.0,79.0,100.0,42.0),
	 ('2024-06-04','II','Machine 3',16672,110,371.0,510.0,55.0,47.97839,73.0,82.0,99.0,59.0),
	 ('2024-06-04','II','Machine 4',7321,19,358.0,510.0,45.0,42.03535,70.0,45.0,100.0,31.0),
	 ('2024-06-04','II','Machine 5',20923,0,277.0,510.0,60.0,55.659714,54.0,126.0,100.0,68.0),
	 ('2024-06-04','III','Machine 1',11783,0,289.0,420.0,52.0,50.442795,69.0,78.0,100.0,53.0),
	 ('2024-06-04','III','Machine 3',6152,35,253.0,420.0,55.0,44.401234,60.0,44.0,99.0,26.0),
	 ('2024-06-04','III','Machine 4',3754,0,313.0,420.0,45.0,41.9373,75.0,27.0,100.0,20.0),
	 ('2024-06-04','III','Machine 5',10430,0,269.0,420.0,60.0,56.28022,64.0,65.0,100.0,41.0),
	 ('2024-06-05','I','Machine 1',1594,0,29.0,510.0,52.0,38.511345,6.0,106.0,100.0,6.0),
	 ('2024-06-05','I','Machine 2',12000,0,283.0,510.0,45.0,41.04241,55.0,94.0,100.0,51.0),
	 ('2024-06-05','I','Machine 3',14061,12,362.0,510.0,55.0,47.449165,71.0,71.0,100.0,50.0),
	 ('2024-06-05','I','Machine 4',11820,20,296.0,510.0,45.0,40.66127,58.0,89.0,100.0,51.0),
	 ('2024-06-05','I','Machine 5',14733,0,280.0,510.0,60.0,53.877296,55.0,88.0,100.0,48.0),
	 ('2024-06-05','II','Machine 1',8971,14,334.0,510.0,52.0,50.66443,65.0,52.0,100.0,33.0),
	 ('2024-06-05','II','Machine 2',15414,0,360.0,510.0,45.0,41.994225,71.0,95.0,100.0,67.0),
	 ('2024-06-05','II','Machine 3',12764,0,304.0,510.0,55.0,46.59375,60.0,76.0,100.0,45.0),
	 ('2024-06-05','II','Machine 4',15320,20,361.0,510.0,45.0,41.84378,71.0,94.0,100.0,66.0),
	 ('2024-06-06','I','Machine 1',5714,0,279.0,510.0,52.0,48.943127,55.0,39.0,100.0,21.0),
	 ('2024-06-06','I','Machine 2',13700,12,322.0,510.0,45.0,41.15202,63.0,95.0,100.0,59.0),
	 ('2024-06-06','I','Machine 3',17700,0,360.0,510.0,55.0,42.437283,71.0,89.0,100.0,63.0),
	 ('2024-06-06','I','Machine 4',9442,9,259.0,510.0,45.0,41.88091,51.0,81.0,100.0,41.0),
	 ('2024-06-06','I','Machine 5',18860,0,324.0,510.0,60.0,56.92071,64.0,97.0,100.0,62.0),
	 ('2024-06-06','III','Machine 1',13019,0,257.0,420.0,52.0,50.422333,61.0,97.0,100.0,59.0),
	 ('2024-06-06','III','Machine 2',10127,0,234.0,420.0,45.0,41.454243,56.0,96.0,100.0,53.0),
	 ('2024-06-06','III','Machine 3',13818,10,295.0,420.0,55.0,46.577343,70.0,85.0,100.0,59.0),
	 ('2024-06-06','III','Machine 4',13031,0,306.0,420.0,45.0,42.173347,73.0,95.0,100.0,69.0),
	 ('2024-06-06','III','Machine 5',15671,0,307.0,420.0,60.0,55.398243,73.0,85.0,100.0,62.0),
	 ('2024-06-07','I','Machine 1',13776,0,267.0,510.0,52.0,49.68332,52.0,99.0,100.0,51.0),
	 ('2024-06-07','I','Machine 2',12095,19,285.0,510.0,45.0,40.853893,56.0,94.0,100.0,52.0),
	 ('2024-06-07','I','Machine 3',18264,40,352.0,510.0,55.0,43.929173,69.0,94.0,100.0,64.0),
	 ('2024-06-07','I','Machine 4',11607,14,266.0,510.0,45.0,33.925346,52.0,97.0,100.0,50.0),
	 ('2024-06-07','I','Machine 5',23540,0,370.0,510.0,60.0,56.678844,73.0,106.0,100.0,77.0),
	 ('2024-06-07','II','Machine 1',17849,30,347.0,510.0,52.0,50.684147,68.0,99.0,100.0,67.0),
	 ('2024-06-07','II','Machine 2',7105,19,333.0,510.0,45.0,42.271317,65.0,47.0,100.0,30.0),
	 ('2024-06-07','II','Machine 3',16662,20,334.0,510.0,55.0,45.206707,65.0,91.0,100.0,59.0),
	 ('2024-06-07','II','Machine 4',16105,14,379.0,510.0,45.0,41.572105,74.0,94.0,100.0,69.0),
	 ('2024-06-07','II','Machine 5',1428,0,266.0,510.0,60.0,54.936234,52.0,9.0,100.0,4.0),
	 ('2024-06-09','I','Machine 1',13060,20,254.0,510.0,52.0,50.530624,50.0,99.0,100.0,49.0),
	 ('2024-06-09','I','Machine 2',15109,0,355.0,510.0,45.0,42.025196,70.0,95.0,100.0,66.0),
	 ('2024-06-09','I','Machine 3',14680,150,355.0,510.0,55.0,45.667007,70.0,75.0,99.0,51.0),
	 ('2024-06-09','I','Machine 4',10701,0,309.0,510.0,45.0,41.741375,61.0,77.0,100.0,46.0),
	 ('2024-06-09','I','Machine 5',11749,0,299.0,510.0,60.0,55.08503,59.0,65.0,100.0,38.0),
	 ('2024-06-09','II','Machine 1',19510,0,379.0,510.0,52.0,51.052567,74.0,99.0,100.0,73.0),
	 ('2024-06-09','II','Machine 2',15266,76,361.0,510.0,45.0,41.929104,71.0,94.0,100.0,66.0),
	 ('2024-06-09','II','Machine 3',19686,0,330.0,510.0,55.0,44.594093,65.0,108.0,100.0,70.0),
	 ('2024-06-09','II','Machine 4',18025,0,378.0,510.0,45.0,42.195587,74.0,106.0,100.0,78.0),
	 ('2024-06-09','II','Machine 5',12720,0,360.0,510.0,60.0,55.837994,71.0,59.0,100.0,41.0),
	 ('2024-06-09','III','Machine 1',11202,0,217.0,420.0,52.0,50.488136,52.0,99.0,100.0,51.0),
	 ('2024-06-09','III','Machine 2',7350,63,175.0,420.0,45.0,38.822624,42.0,93.0,99.0,38.0),
	 ('2024-06-09','III','Machine 3',15400,5,311.0,420.0,55.0,44.64419,74.0,90.0,100.0,66.0),
	 ('2024-06-09','III','Machine 4',13792,6,314.0,420.0,45.0,42.231377,75.0,98.0,100.0,73.0),
	 ('2024-06-09','III','Machine 5',22492,0,240.0,420.0,60.0,56.529236,57.0,156.0,100.0,88.0),
	 ('2024-06-10','I','Machine 1',15140,25,288.0,510.0,52.0,50.513477,56.0,101.0,100.0,56.0),
	 ('2024-06-10','I','Machine 2',15502,71,360.0,510.0,45.0,42.164093,71.0,96.0,100.0,68.0),
	 ('2024-06-10','I','Machine 3',15425,390,312.0,510.0,55.0,47.27796,61.0,90.0,98.0,53.0),
	 ('2024-06-10','I','Machine 4',14820,6,375.0,510.0,45.0,41.53826,74.0,88.0,100.0,65.0),
	 ('2024-06-10','I','Machine 5',17019,0,289.0,510.0,60.0,56.612553,57.0,98.0,100.0,55.0),
	 ('2024-06-10','II','Machine 1',19505,0,376.0,510.0,52.0,51.217384,74.0,100.0,100.0,74.0),
	 ('2024-06-10','II','Machine 2',13003,38,377.0,510.0,45.0,42.224415,74.0,77.0,100.0,56.0),
	 ('2024-06-10','II','Machine 3',15286,50,360.0,510.0,55.0,48.625023,71.0,77.0,100.0,54.0),
	 ('2024-06-10','II','Machine 4',15385,0,337.0,510.0,45.0,42.069523,66.0,101.0,100.0,66.0),
	 ('2024-06-10','II','Machine 5',19539,0,336.0,510.0,60.0,56.97629,66.0,97.0,100.0,64.0),
	 ('2024-06-10','III','Machine 1',16128,0,312.0,420.0,52.0,50.9542,74.0,99.0,100.0,73.0),
	 ('2024-06-10','III','Machine 2',16297,11,302.0,420.0,45.0,41.986385,72.0,120.0,100.0,86.0),
	 ('2024-06-10','III','Machine 3',12351,30,196.0,420.0,55.0,46.07515,47.0,115.0,100.0,54.0),
	 ('2024-06-10','III','Machine 4',8507,13,191.0,420.0,45.0,40.682995,45.0,99.0,100.0,44.0),
	 ('2024-06-10','III','Machine 5',15820,0,273.0,420.0,60.0,55.946102,65.0,97.0,100.0,63.0);