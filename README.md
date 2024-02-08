### Source for Body Condition Score Assertment of Dairy Cow using Deep Learning and 3D Imaging

#### 1. Overall process
![image info](./media/overall.png)


### 2. Compute POV feature
Refer to code file <span style="color:blue">make_view.py</span>. and make_full_view in folder scripts
![image info](./media/pov.png)

### 3. Training
```
python train_pointnet_with_pov.py
```

### 3. Scoring
```
python scoring_by_pov.py
```
