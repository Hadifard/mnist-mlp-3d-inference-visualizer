# Walkthrough video

This folder is the place for the project walkthrough (2 to 3 minutes) that shows how to use the application.

## Add the video

1. Copy the video into this folder and name it exactly:

   ```
   docs/media/movie_how_it_works.mp4
   ```

   (Use underscores instead of spaces so the link in the README works everywhere.)

2. Commit and push. The poster image in the main README already links to this file:

   ```markdown
   [![Watch the walkthrough](docs/images/video-poster.png)](docs/media/movie_how_it_works.mp4)
   ```

## Notes

- GitHub previews MP4 files up to 100 MB when you open them, and a normal push accepts files up to 100 MB. For a 2 to 3 minute screen recording, 1080p at a moderate bitrate stays well below this.
- If the file is bigger, re-encode it (for example with `ffmpeg -i input.mp4 -vcodec libx264 -crf 28 movie_how_it_works.mp4`) or upload it to a video platform and replace the link in the README.
- To play the video directly inside the README, open the README in the GitHub web editor and drag the MP4 file into it. GitHub then creates an inline player link that you can paste under the poster.

## Suggested outline for the video

| Time | Content |
|------|---------|
| 0:00 | The screen: drawing grid, 3D network, confidence chart |
| 0:20 | Draw a digit and read the prediction |
| 0:50 | Rotate, zoom and use the camera widget |
| 1:15 | Silk Glow mode and connection settings |
| 1:45 | Click a neuron: weights, contributions, `z` and activation |
| 2:15 | Neuron analysis panel |
| 2:40 | Training timeline |
