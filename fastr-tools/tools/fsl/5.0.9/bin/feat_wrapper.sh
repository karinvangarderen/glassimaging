#!/bin/sh

function edit_design {
KEY="$1"; VALUE=$2; DESIGN="$3"
if [ -n "$VALUE" ]; then
  echo $KEY -\> $VALUE
  sed -i "/$KEY/c $KEY $VALUE" $DESIGN
else
  echo $KEY -\> No value found
fi
}

# Parse the input arguments
if [ -n "$2" ]; then
  DESIGN="$1"
  FMRI="$2"
else
  echo -e "\nusage: feat_wrapper.sh design.fsf fmri_images [output_directory [TR [t1_image_brain t1_image]]]\n"
  exit 1
fi
FEAT_DIR=$3
# Try reading TR, TE and the dwelltime values from the nifti-header
eval $(fslhd "$FMRI" | grep descrip | sed -r 's/[[:alnum:]]+=/\n&/g' | tr '[:lower:]' '[:upper:]' | grep 'TR=\|TE=\|DWELL=')
if [ -n "$4" ]; then
  TR=$4
fi
T1=$5
T1_ORIG=$6

# Copy the master design/setup-file
DESIGN_EDIT=$(dirname "$FMRI")/$(basename "$FMRI" | cut -d. -f1)_$(basename "$DESIGN")
echo $DESIGN --\> $DESIGN_EDIT
cp "$DESIGN" "$DESIGN_EDIT"

# Edit the subject-specific key-value pairs in the design
edit_design 'set feat_files(1)'   "\"$(echo "$FMRI" | cut -f 1 -d '.')\"" "$DESIGN_EDIT"
edit_design 'set fmri(npts)'      "$(fslnvols "$FMRI")"                   "$DESIGN_EDIT"
edit_design 'set fmri(tr)'        "$TR"                                   "$DESIGN_EDIT"
edit_design 'set fmri(te)'        "$TE"                                   "$DESIGN_EDIT"
edit_design 'set fmri(dwell)'     "$DWELL"                                "$DESIGN_EDIT"
edit_design 'set fmri(outputdir)' "\"$FEAT_DIR\""                         "$DESIGN_EDIT"

# Create a symbolic link with a *_brain suffix to make sure FEAT can find the original T1
BASENAME="$(basename $T1_ORIG)"
FILENAME="${BASENAME%%.*}"
FILE_EXT="${BASENAME#*.}"
LINKNAME="$(dirname $T1_ORIG)/${FILENAME}_brain.${FILE_EXT}"
if [ -n "$T1_ORIG" ] && [ "$T1" != "$LINKNAME" ]; then
  ln -s "$T1" "$LINKNAME"
  T1="$LINKNAME"
fi

# Edit or remove the T1 key-value pair in the design
BASENAME="$(basename $T1)"
FILENAME="${BASENAME%%.*}"
HEADER="# Subject's structural image for analysis 1"
T1_KEY="set highres_files(1)"
T1_VALUE="\"$(dirname $T1)/$FILENAME\""
if [ -n "$T1" ]; then
  if [ -n "$(grep "$T1_KEY" "$DESIGN_EDIT")" ]; then
    edit_design "$T1_KEY" "$T1_VALUE" "$DESIGN_EDIT"
  else
    echo $T1_KEY -\> $T1_VALUE
    echo -e "\n##########################################################\n\n$HEADER\n$T1_KEY $T1_VALUE\n" >> "$DESIGN_EDIT"
  fi
else
  sed -i -e "/$HEADER/d" -e "/$T1_KEY/d" "$DESIGN_EDIT"
fi

# Run FEAT with the edited design
echo Running: feat $DESIGN_EDIT
feat "$DESIGN_EDIT"
