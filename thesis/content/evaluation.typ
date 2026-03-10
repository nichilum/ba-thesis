#import "/thesis/utils/todo.typ": TODO

= Evaluation

== DISCUSS UPSAMPLING FOR TRAINING AND REVERBERATING AT LOWER (USING PLOTS)<disc_upsampling>

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model

== (An)echoic dataset
- AudioSet and FSD50K are not "dry" datasets. they contain samples that are recorded under echoic conditions. 
- The model is not shown fully dereverberated sample pairs during training