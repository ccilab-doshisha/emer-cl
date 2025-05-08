import os

import numpy as np
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

BASE_: str = os.path.join(os.environ["WORKDIR"], "src", "data")
BASE: str = os.path.join(BASE_, "PMEmo")
DATASET: str = os.path.join(BASE, "dataset", "PMEmo2019")
AUDIO: str = os.path.join(DATASET, "chorus")


def discard() -> None:

    if os.path.isfile(
        os.path.join(DATASET, "metadata_discard.csv")
    ) and os.path.isfile(os.path.join(DATASET, "remain_music_indices.npy")):
        return
    original_discard_time = 15.0 + 0.5

    # check the discard up to 25s, it can use
    df_metadata = pd.read_csv(
        os.path.join(DATASET, "metadata.csv"), header=0, index_col="musicId"
    )

    # Samples under 15.5s were discarded in advance.
    original_num = len(df_metadata)
    print("\033[1;32m[LOG]\033[0m original (did not discard) :", original_num)

    # Ensure that it exists for at least 7 seconds
    discard_time = 7.0
    discarded_df = df_metadata[
        df_metadata["duration"] >= original_discard_time + discard_time
    ]
    discarded_num = len(discarded_df)
    print("\033[1;32m[LOG]\033[0m discard :", discarded_num)
    # metadata_save
    discarded_df.to_csv(os.path.join(DATASET, "metadata_discard.csv"))
    remain_music_indices = discarded_df.index
    np.save(os.path.join(BASE, "remain_music_indices"), remain_music_indices)


def main() -> None:

    remain_music_indices = np.load(
        os.path.join(BASE, "remain_music_indices.npy")
    )

    save_dir = os.path.join(AUDIO, "wav")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len([f.name for f in os.scandir(save_dir) if f.is_file()]) == len(
        remain_music_indices
    ):
        return

    # clip [PMEmo is need to discard under 15sec] and convert mp3 to wav
    for idx in tqdm(remain_music_indices):
        sound_orig = AudioSegment.from_file(
            os.path.join(AUDIO, "%d.mp3" % (idx)), "mp3"
        )
        clipped_sound = sound_orig[15 * 1000 :]
        clipped_sound.export(
            os.path.join(save_dir, "audio_%d.wav" % (idx)), format="wav"
        )


if __name__ == "__main__":
    discard()
    main()
