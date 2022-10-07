using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ArcFace{
    public class NNForSimilarity{
        public InferenceSession session;
        public NNForSimilarity () {
            using var modelStream = typeof(NNForSimilarity).Assembly.GetManifestResourceStream("arcfaceresnet.onnx");
            using var memoryStream = new MemoryStream();
            if (modelStream != null)
                modelStream.CopyTo(memoryStream);
            this.session = new InferenceSession(memoryStream.ToArray()); 
        }
        string MetadataToString(NodeMetadata metadata) => $"{metadata.ElementType}[{String.Join(",", metadata.Dimensions.Select(i => i.ToString()))}]";

        float Length(float[] v) => (float)Math.Sqrt(v.Select(x => x*x).Sum());

        float[] Normalize(float[] v) 
        {
            var len = Length(v);
            return v.Select(x => x / len).ToArray();
        }

        float Distance(float[] v1, float[] v2) => Length(v1.Zip(v2).Select(p => p.First - p.Second).ToArray());

        float Similarity(float[] v1, float[] v2) => v1.Zip(v2).Select(p => p.First * p.Second).Sum();
        bool IsCancel(CancellationToken ct){
            if(ct.IsCancellationRequested){
                return true;
            }
            else{
                return false;
            }
        }

        DenseTensor<float> ImageToTensor(Image<Rgb24> img)
        {
            var w = img.Width;
            var h = img.Height;
            var t = new DenseTensor<float>(new[] { 1, 3, h, w });

            img.ProcessPixelRows(pa => 
            {
                for (int y = 0; y < h; y++)
                {           
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        t[0, 0, y, x] = pixelSpan[x].R;
                        t[0, 1, y, x] = pixelSpan[x].G;
                        t[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });

            return t;
        }
        float[] GetEmbeddings(Image<Rgb24> face) 
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", ImageToTensor(face)) };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
            return Normalize(results.First(v => v.Name == "fc1").AsEnumerable<float>().ToArray());
        }
        async Task<float[]> GetEmbeddingsAsync(Image<Rgb24> face, CancellationToken ct) 
        {
            return await Task<float[]>.Factory.StartNew(() =>
            {
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", ImageToTensor(face)) };
                lock(session)
                {
                    using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
                    return Normalize(results.First(v => v.Name == "fc1").AsEnumerable<float>().ToArray());
                }
            }, ct);
            
            
            
            
        }
        public IEnumerable <(string, double)> GetDistanceNSimilarity (Image<Rgb24> face1, Image<Rgb24> face2){
            var L = new List<(string, double)>();
            
            var embeddings1 = GetEmbeddings(face1);
            var embeddings2 = GetEmbeddings(face2);

            L.Add(("Distance =", Distance(embeddings1, embeddings2) * Distance(embeddings1, embeddings2)));
            L.Add(("Similarity =", Similarity(embeddings1, embeddings2)));

            return L;
        }
        
        public async Task<IEnumerable <(string, double)>> AsyncGetDistanceNSimilarity (Image<Rgb24> face1, Image<Rgb24> face2, CancellationToken token, string? taskName = null){
            return await Task<IEnumerable <(string, double)>>.Factory.StartNew(() =>
            {
                var L = new List<(string, double)>();
                
                var embeddings1 = GetEmbeddingsAsync(face1, token);
                if(IsCancel(token)){
                    return L;
                }
                var embeddings2 = GetEmbeddingsAsync(face2, token);

                L.Add(("Distance =", Distance(embeddings1.Result, embeddings2.Result) * Distance(embeddings1.Result, embeddings2.Result)));
                L.Add(("Similarity =", Similarity(embeddings1.Result, embeddings2.Result)));

                return L;
            }, token);
        }
    }
}