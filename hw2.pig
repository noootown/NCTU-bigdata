y = LOAD '*.csv' USING PigStorage(',') AS (
    Year:chararray,
    Month:chararray,
    DayofMonth:chararray,
    DayOfWeek:chararray,
    DepTime:chararray,
    CRSDepTime:chararray,
    ArrTime:chararray,
    CRSArrTime:chararray,
    UniqueCarrier:chararray,
    FlightNum:chararray,
    TailNum:chararray,
    ActualElapsedTime:chararray,
    CRSElapsedTime:chararray,
    AirTime:chararray,
    ArrDelay:int,
    DepDelay:int,
    Origin:chararray,
    Dest:chararray,
    Distance:chararray,
    TaxiIn:chararray,
    TaxiOut:chararray,
    Cancelled:chararray,
    CancellationCode:chararray,
    Diverted:chararray,
    CarrierDelay:int,
    WeatherDelay:int,
    NASDelay:int,
    SecurityDelay:int,
    LateAircraftDelay:int);

y = FILTER y BY Year != 'Year';

-- Q1
grouped = GROUP y all;
delay = FOREACH grouped GENERATE AVG(y.DepDelay), AVG(y.ArrDelay);
dump delay;

y_m = GROUP y BY (Month);
y_m_maxdelay = FOREACH y_m GENERATE group, MAX(y.DepDelay), MAX(y.ArrDelay);
y_m_maxdelay = ORDER y_m_maxdelay BY $0 ASC;
dump y_m_maxdelay

-- Q2
y_weather = FILTER y BY WeatherDelay > 0;
y_weather = GROUP y_weather ALL;
y_weather_delay = FOREACH y_weather GENERATE COUNT(y_weather), AVG(y_weather.WeatherDelay);
dump y_weather_delay;

-- Q3
y_m = GROUP y BY (Month);
y_m_delay = FOREACH y_m GENERATE group, AVG(y.ArrDelay);
dump y_m_delay

-- Q4
y_ori = GROUP y BY (Origin);
y_ori_delay = FOREACH y_ori GENERATE group, AVG(y.DepDelay), AVG(y.CarrierDelay);
y_ori_delay = ORDER y_ori_delay BY $1 DESC;
y_ori_delay = LIMIT y_ori_delay 5;
dump y_ori_delay;

y_des = GROUP y BY (Dest);
y_des_delay = FOREACH y_des GENERATE group, AVG(y.ArrDelay);
y_des_delay = ORDER y_des_delay BY $1 DESC;
y_des_delay = LIMIT y_des_delay 5;
dump y_des_delay;

