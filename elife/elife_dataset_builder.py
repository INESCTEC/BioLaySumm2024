"""elife dataset."""

import tensorflow_datasets as tfds
import os
import json

# Define the file paths for your train, validation, and test sets
TRAIN_FILEPATH = 'eLife_train.jsonl'
VAL_FILEPATH = 'eLife_val.jsonl'
TEST_FILEPATH = 'eLife_test.jsonl'


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for elife dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Detailed download instructions (which require running a custom script) are
  here: https://www.codabench.org/competitions/1920/. Extract biolaysumm2024_data.zip.
  Afterwards, please put eLife* files in the manual_dir.
  """

  def _info(self) -> tfds.core.DatasetInfo:
      """Returns the dataset metadata."""
      return self.dataset_info_from_configs(
          features=tfds.features.FeaturesDict({
              # These are the features of your dataset 
              'lay_summary': tfds.features.Text(),
              'article': tfds.features.Text(),
              'headings': tfds.features.Sequence(tfds.features.Text()),
              'keywords': tfds.features.Sequence(tfds.features.Text())
              #'id': tfds.features.Text()
          }),

          # Typically this is a (input_key, target_key) tuple, and the dataset 
          # yields a tuple of tensors (input, target) tensors.
          # Since we have to pass article, heading and keywords as input, then 
          # it is None
          supervised_keys=None,  # Set to `None` to disable
      )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
      """Returns SplitGenerators."""
    
      # Yield the splits along with their corresponding file paths
      return {
        'train': self._generate_examples(TRAIN_FILEPATH, image_base_path=dl_manager.manual_dir),
        'validation': self._generate_examples(VAL_FILEPATH, image_base_path=dl_manager.manual_dir)
        #'test': self._generate_examples(TEST_FILEPATH, image_base_path=dl_manager.manual_dir),
      }

  def _generate_examples(self, filepath, image_base_path):
      # Open the JSONL file and yield examples
      with open(os.path.join(image_base_path, filepath), 'r',encoding='utf-8') as f:
                # for i, line in enumerate(f):
                for line in f:
                    data = json.loads(line)
                    yield data['id'] , {
                        'lay_summary': data['lay_summary'],
                        'article': data['article'],
                        'headings': data['headings'],
                        'keywords': data['keywords']
                    }

