# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

set -e

cd src

while getopts ":pdf" opt; do
  case $opt in
    p)
      echo "installing packages"

      cd leap
      pip install -e .

      cd ../maml
      pip install -e .
      
      cd ..
      echo "done"     
      ;;
    d)
      echo "Getting data"
      
      cd omniglot
      bash getdata.sh
      
      cd ..
      echo "done"
      ;;
    f)
      echo "making log dirs"
      cd omniglot

      mkdir -p logs
      mkdir -p logs/leap
      mkdir -p logs/reptile
      mkdir -p logs/maml
      mkdir -p logs/fomaml
      mkdir -p logs/ft
      mkdir -p logs/no
      echo "done"

      cd ..
      cd ..

      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done
