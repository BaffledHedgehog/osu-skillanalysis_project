using Realms;
using System.Text.Json;

// Open only a caller-created snapshot, never the live client.realm.
if (args.Length != 1 || !File.Exists(args[0]))
{
    Console.Error.WriteLine("Expected path to a client.realm snapshot.");
    return 2;
}

try
{
    var config = new RealmConfiguration(Path.GetFullPath(args[0])) { IsReadOnly = true, IsDynamic = true };
    using var realm = Realm.GetInstance(config);
    bool Has(IRealmObjectBase obj, string name) => obj.ObjectSchema?.Any(p => p.Name == name) == true;
    T? Get<T>(IRealmObjectBase? obj, string name) => obj is not null && Has(obj, name) ? obj.DynamicApi.Get<T>(name) : default;
    var result = new List<object>();
    if (realm.Schema.Any(s => s.Name == "Score"))
    {
        foreach (IRealmObjectBase score in realm.DynamicApi.All("Score"))
        {
            if (result.Count >= 500000) break;
            if (Get<bool>(score, "DeletePending")) continue;
            var user = Get<IRealmObjectBase>(score, "User");
            var beatmap = Get<IRealmObjectBase>(score, "BeatmapInfo");
            var ruleset = Get<IRealmObjectBase>(score, "Ruleset");
            var files = new List<object>();
            if (Has(score, "Files"))
            {
                foreach (var usage in score.DynamicApi.GetList<IRealmObjectBase>("Files"))
                {
                    var file = Get<IRealmObjectBase>(usage, "File");
                    files.Add(new { name = Get<string>(usage, "Filename"), hash = Get<string>(file, "Hash") });
                }
            }
            result.Add(new {
                id = Get<Guid>(score, "ID"), user = Get<string>(user, "Username"),
                mode = Get<long>(ruleset, "OnlineID"), accuracy = Get<double>(score, "Accuracy"),
                online_id = Get<long>(score, "OnlineID"), legacy_online_id = Get<long>(score, "LegacyOnlineID"),
                played_at = Get<DateTimeOffset>(score, "Date"), statistics = Get<string>(score, "Statistics"),
                rank = Has(score, "Rank") ? (long?)Get<long>(score, "Rank") : null,
                is_legacy = Get<bool>(score, "IsLegacyScore"),
                mods = Get<string>(score, "Mods"), map_hash = Get<string>(score, "BeatmapHash"),
                map_file_hash = Get<string>(beatmap, "Hash"), map_online_id = Get<long>(beatmap, "OnlineID"),
                replay_files = files
            });
        }
    }
    Console.Write(JsonSerializer.Serialize(new { scores = result }));
    return 0;
}
catch (Exception)
{
    Console.Error.WriteLine("Cannot read this Realm snapshot. Close osu!lazer and retry, or update LazerReader.");
    return 1;
}
