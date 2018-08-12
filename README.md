## Structure
```
data: input
app: program
output/model: selection result and model file
output/prediction: prediction result
output/report: simulation performance
```

## Data format
```buildoutcfg
1. columns: date,feature1,feature2,...label1,..label4
2. using number as input
3. label will convert to class comparing to 0 (>0: positive)

```

## Build
make install

## Run
make predict

## Reset model
make reset


