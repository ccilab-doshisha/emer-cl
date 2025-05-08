import os

from pydub import AudioSegment
from tqdm import tqdm

BASE_: str = os.path.join(os.environ["WORKDIR"], "src", "data")
BASE: str = os.path.join(BASE_, "DEAM")
DATASET: str = os.path.join(BASE, "dataset")
AUDIO: str = os.path.join(DATASET, "MEMD_audio")


def main() -> None:

    file_nums = sorted(
        [
            int(f.name.split(".")[0])
            for f in os.scandir(AUDIO)
            if f.name.endswith("mp3")
        ]
    )
    save_dir = os.path.join(AUDIO, "wav")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(
        [f.name for f in os.scandir(save_dir) if f.name.endswith("wav")]
    ) == len(file_nums):
        return

    # clip [DEAM is need to clip 15 ~ 45 sec] and convert mp3 to wav
    for num in tqdm(file_nums):
        sound_orig = AudioSegment.from_file(
            os.path.join(AUDIO, "%d.mp3" % (num)), "mp3"
        )

        clipped_sound = sound_orig[15 * 1000 : 45 * 1000]
        if clipped_sound.duration_seconds < 30:
            """
            NOTE:
            We have found that there are some audio records that do not satisfy the 30s.
            (e.g. #146, #272, #435, #990, #1174, #1200, #1273, #1493 and #1789)
            Therefore, for those records,
            we inserted a little silence at the end of the audio.
            This ensures we can extract VGGish features for every 30s in all records.
            """
            silent_pad_sec: int = 30 - clipped_sound.duration_seconds
            clipped_sound = clipped_sound + AudioSegment.silent(
                duration=silent_pad_sec * 1000
            )

        clipped_sound.export(
            os.path.join(save_dir, "audio_%d.wav" % (num)), format="wav"
        )


if __name__ == "__main__":
    main()
