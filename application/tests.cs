using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using ArcFace;
using System.Diagnostics;

internal class Program
{
    private static void Main(string[] args){
        timeTest();
        //CancelTest();
    }

    public static void timeTest()
    {
        CancellationTokenSource cts = new CancellationTokenSource();
        CancellationToken ct = cts.Token;

        var images = new List<Image<Rgb24>>();
        for (int i = 1; i <= 8; ++i)
        {
            images.Add(Image.Load<Rgb24>($"face{i}.png"));
        }

        NNForSimilarity ArcF = new NNForSimilarity();
        Stopwatch stopWatch = new Stopwatch();
        stopWatch.Start();
        for (int i = 0; i <= 7; i += 2)
        {
            var L = ArcF.GetDistanceNSimilarity(images[i], images[i + 1]);
            foreach (var item in L)
            {
                Console.WriteLine($"{item.Item1}: {item.Item2}");
            }
        }
        stopWatch.Stop();
        Console.WriteLine("Sync" + stopWatch.ElapsedMilliseconds);

        stopWatch.Reset();
        stopWatch.Start();
        Task[] Ls = new Task[4];
        for (int i = 0; i < Ls.Length; i++)
        {
            Ls[i] = ArcF.AsyncGetDistanceNSimilarity(images[i * 2], images[i * 2 + 1], ct);
        }
        Task.WaitAll(Ls);
        for (int i = 0; i < Ls.Length; i++)
        {
            var L = (Task<IEnumerable<(string, double)>>)Ls[i];
            foreach (var item in L.Result)
            {
                Console.WriteLine($"{item.Item1}: {item.Item2}");
            }
        }
        stopWatch.Stop();
        Console.WriteLine("Async" + stopWatch.ElapsedMilliseconds);
    }
    public static void CancelTest(){
        CancellationTokenSource cts = new CancellationTokenSource();
        CancellationToken ct = cts.Token;

        var images = new List<Image<Rgb24>>();
        for (int i = 1; i <= 8; ++i)
        {
            images.Add(Image.Load<Rgb24>($"face{i}.png"));
        }

        NNForSimilarity ArcF = new NNForSimilarity();
        Task[] Ls = new Task[4];
        cts.Cancel();
        for (int i = 0; i < Ls.Length; i++)
        {
            Ls[i] = ArcF.AsyncGetDistanceNSimilarity(images[i * 2], images[i * 2 + 1], ct);
        }
        Task.WaitAll(Ls);
        for (int i = 0; i < Ls.Length; i++)
        {
            var L = (Task<IEnumerable<(string, double)>>)Ls[i];
            foreach (var item in L.Result)
            {
                Console.WriteLine($"{item.Item1}: {item.Item2}");
            }
        }
    }
}