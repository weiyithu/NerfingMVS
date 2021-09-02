SCENE=$1
sh colmap.sh data/$SCENE
python run.py --config configs/$SCENE.txt  --no_ndc --spherify --lindisp --demo  --expname=$SCENE
