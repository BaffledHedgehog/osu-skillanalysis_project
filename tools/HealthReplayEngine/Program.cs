using System.Security.Cryptography;
using System.Text.Json;
using osu.Framework.Allocation;
using osu.Framework.Audio;
using osu.Framework.Configuration;
using osu.Framework.Platform;
using osu.Game;
using osu.Game.Beatmaps;
using osu.Game.Beatmaps.Formats;
using osu.Game.IO;
using osu.Game.Rulesets;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Objects;
using osu.Game.Scoring;
using osu.Game.Scoring.Legacy;
using osu.Game.Tests.Visual;

bool audioCheck = args.Length == 2 && args[0] == "--audio-check";
if (!audioCheck && (args.Length is not (3 or 4) || args.Length == 4 && args[3] != "--events")) { Console.Error.WriteLine("Expected beatmap.osu replay.osr result.json [--events], or --audio-check result.json"); return 2; }
string outputPath = Path.GetFullPath(args[audioCheck ? 1 : 2]);
using var host = new SilentReplayHost("pulse-health-" + Guid.NewGuid().ToString("N"));
host.Run(new ReplayGame(audioCheck ? "" : Path.GetFullPath(args[0]), audioCheck ? "" : Path.GetFullPath(args[1]), outputPath, args.Length == 4, audioCheck));
return File.Exists(outputPath) ? 0 : 1;

// Headless rendering does not guarantee silent audio after game configuration
// bindings are loaded. Force silence both before and after those bindings.
public class SilentReplayHost(string name) : HeadlessGameHost(name, realtime: false)
{
    protected override void SetupConfig(IDictionary<FrameworkSetting, object> defaultOverrides)
    {
        defaultOverrides[FrameworkSetting.VolumeUniversal] = 0.0;
        defaultOverrides[FrameworkSetting.VolumeEffect] = 0.0;
        defaultOverrides[FrameworkSetting.VolumeMusic] = 0.0;
        base.SetupConfig(defaultOverrides);
        Config.SetValue(FrameworkSetting.AudioDevice, "No sound");
        Config.SetValue(FrameworkSetting.VolumeUniversal, 0.0);
        Config.SetValue(FrameworkSetting.VolumeEffect, 0.0);
        Config.SetValue(FrameworkSetting.VolumeMusic, 0.0);
    }
}

public partial class ReplayGame(string map, string replay, string result, bool collectEvents, bool audioCheck) : OsuGameBase
{
    protected override void LoadComplete()
    {
        base.LoadComplete();
        var AudioManager = Audio;
        // Keep these at zero even if the game's settings or a test scene tries
        // to restore a default volume. No playback scene exists at this point.
        AudioManager.Volume.BindValueChanged(e => { if (e.NewValue != 0) AudioManager.Volume.Value = 0; }, true);
        AudioManager.VolumeSample.BindValueChanged(e => { if (e.NewValue != 0) AudioManager.VolumeSample.Value = 0; }, true);
        AudioManager.VolumeTrack.BindValueChanged(e => { if (e.NewValue != 0) AudioManager.VolumeTrack.Value = 0; }, true);
        AudioManager.AudioDevice.Value = "No sound";
        if (audioCheck)
        {
            // Exercise attempted restoration as well, without loading a replay
            // or playing a sample. The guards must immediately force zero.
            AudioManager.Volume.Value = 1;
            AudioManager.VolumeSample.Value = 1;
            AudioManager.VolumeTrack.Value = 1;
            bool muted = AudioManager.Volume.Value == 0 && AudioManager.VolumeSample.Value == 0 && AudioManager.VolumeTrack.Value == 0;
            File.WriteAllText(result, JsonSerializer.Serialize(new {
                status = muted ? "verified" : "audio_not_muted", audio_muted = muted,
                universal_volume = AudioManager.Volume.Value, sample_volume = AudioManager.VolumeSample.Value,
                track_volume = AudioManager.VolumeTrack.Value, device = AudioManager.AudioDevice.Value,
                no_replay_loaded = true
            }));
            Exit();
            return;
        }
        var scene = new HealthScene(map, replay, result, collectEvents);
        scene.OnLoadComplete += _ => scene.RunAllSteps(Exit, (_, ex) => {
            File.WriteAllText(result, JsonSerializer.Serialize(new { status = "error", error = ex.ToString() }));
            Exit();
        });
        Add(scene);
    }
}

public partial class HealthScene(string mapPath, string replayPath, string resultPath, bool collectEvents) : ScreenTestScene
{
    private TestReplayPlayer? player;
    private Score? score;
    private readonly List<object> points = [];
    private readonly List<object> events = [];
    private readonly Dictionary<HitObject, int> rootIndices = [];
    private readonly HashSet<int> observedRoots = [];
    private int rootCount;
    private int unmappedEvents;
    private bool revertedEvents;
    private double? failTime;
    private double minimum = 1;
    private double lastSample = double.NegativeInfinity;
    protected override Ruleset CreateRuleset() => new OsuRuleset();
    protected override bool UseFreshStoragePerRun => true;

    [BackgroundDependencyLoader]
    private void load()
    {
        AddStep("load exact beatmap and replay", () => {
            using var reader = new LineBufferedReader(File.OpenRead(mapPath));
            var beatmap = Decoder.GetDecoder<Beatmap>(reader).Decode(reader);
            beatmap.BeatmapInfo.MD5Hash = Convert.ToHexString(MD5.HashData(File.ReadAllBytes(mapPath))).ToLowerInvariant();
            Beatmap.Value = CreateWorkingBeatmap(beatmap);
            Ruleset.Value = new OsuRuleset().RulesetInfo;
            using var stream = File.OpenRead(replayPath);
            score = new LocalDecoder(Beatmap.Value).Parse(stream);
            if (!collectEvents && !score.ScoreInfo.Mods.Any(mod => mod.Acronym == "NF"))
                throw new InvalidDataException("This analysis requires a No Fail replay");
            SelectedMods.Value = score.ScoreInfo.Mods;
            player = new TestReplayPlayer(score, allowPause: false, showResults: false);
            player.OnLoadComplete += _ => {
                player.HealthProcessor.Failed += () => {
                    failTime ??= player.GameplayClockContainer.CurrentTime;
                    return false;
                };
                if (collectEvents)
                {
                    int index = 0;
                    foreach (var obj in player.DrawableRuleset.Objects) registerObject(obj, index++);
                    rootCount = index;
                    player.ScoreProcessor.JudgementReverted += _ => revertedEvents = true;
                    player.ScoreProcessor.NewJudgement += judgement => {
                        if (!rootIndices.TryGetValue(judgement.HitObject, out int root)) { unmappedEvents++; return; }
                        observedRoots.Add(root);
                        events.Add(new {
                            root_index = root, object_type = judgement.HitObject.GetType().Name,
                            object_time_ms = judgement.HitObject.StartTime,
                            time_ms = judgement.TimeAbsolute, offset_ms = judgement.TimeOffset,
                            result = judgement.Type.ToString().ToLowerInvariant(), is_hit = judgement.IsHit,
                            combo_before = judgement.ComboAtJudgement, combo_after = judgement.ComboAfterJudgement,
                            hp_before = judgement.HealthAtJudgement
                        });
                    };
                }
            };
            LoadScreen(player);
        });
        AddUntilStep("wait for player", () => player?.IsLoaded == true);
        AddUntilStep("wait for replay completion", () => player?.GameplayState.HasCompleted == true);
        AddStep("write HP result", () => {
            var statistics = player!.ScoreProcessor.Statistics.ToDictionary(p => p.Key.ToString().ToLowerInvariant(), p => p.Value);
            var expected = score!.ScoreInfo.Statistics.Where(p => p.Value != 0).ToDictionary(p => p.Key.ToString().ToLowerInvariant(), p => p.Value);
            bool matches = statistics.Where(p => p.Value != 0).OrderBy(p => p.Key).SequenceEqual(expected.OrderBy(p => p.Key));
            File.WriteAllText(resultPath, JsonSerializer.Serialize(new {
                status = matches ? "verified" : "statistics_mismatch", engine_version = typeof(OsuGameBase).Assembly.GetName().Version?.ToString(),
                health_processor = player.HealthProcessor.GetType().Name, time_basis = "beatmap_ms",
                statistics_match = matches,
                replay_client_version = score.ScoreInfo.ClientVersion,
                without_nf_fail_time_ms = failTime, minimum_hp = minimum, hp = points,
                statistics, original_statistics = score!.ScoreInfo.Statistics.ToDictionary(p => p.Key.ToString().ToLowerInvariant(), p => p.Value),
                mods = score.ScoreInfo.Mods.Select(m => m.Acronym).ToArray(),
                replay_end_time_ms = score.Replay.Frames.LastOrDefault()?.Time
                , events = collectEvents ? events : null,
                events_schema = collectEvents ? "osu-judgements-v1" : null,
                events_verified = collectEvents && matches && unmappedEvents == 0 && !revertedEvents && observedRoots.Count == rootCount,
                root_count = rootCount, observed_root_count = observedRoots.Count, unmapped_events = unmappedEvents,
                reverted_events = revertedEvents
            }));
        });
    }

    private void registerObject(HitObject obj, int root)
    {
        rootIndices[obj] = root;
        foreach (var nested in obj.NestedHitObjects) registerObject(nested, root);
    }

    protected override void Update()
    {
        base.Update();
        if (player?.IsLoaded != true || player.GameplayClockContainer == null) return;
        double time = player.GameplayClockContainer.CurrentTime;
        double hp = player.HealthProcessor.Health.Value;
        minimum = Math.Min(minimum, hp);
        if (time >= 0 && time - lastSample >= 100)
        {
            points.Add(new { time_ms = Math.Round(time, 2), hp = Math.Round(hp, 6) });
            lastSample = time;
        }
    }

    private class LocalDecoder(WorkingBeatmap beatmap) : LegacyScoreDecoder
    {
        protected override Ruleset GetRuleset(int rulesetId) => rulesetId == 0 ? new OsuRuleset() : throw new InvalidDataException("osu!standard only");
        protected override WorkingBeatmap GetBeatmap(string md5Hash)
        {
            if (!string.Equals(md5Hash, beatmap.BeatmapInfo.MD5Hash, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Replay beatmap checksum does not match");
            return beatmap;
        }
    }
}
