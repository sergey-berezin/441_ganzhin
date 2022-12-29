using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Ookii.Dialogs.Wpf;
using System.Threading;
using ArcFace;
using ImageControl = System.Windows.Controls.Image;
using System.IO;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Collections;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using System.Linq.Expressions;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Lab2vb1
{
    public partial class MainWindow : Window
    {
        object obj = new object();
        private string FFolderPath = "";
        private string SFolderPath = "";
        public MainWindow()
        {
            InitializeComponent();
            CancellationButton.IsEnabled = false;
            FirstDeleteButton.IsEnabled = false;
            SecondDeleteButton.IsEnabled = false;
            CalculateButton.IsEnabled = false;
        }
        private readonly NNForSimilarity arcface = new();

        private CancellationTokenSource cts;

        private VistaFolderBrowserDialog folderDialog = new();

        private void FolderButton_Click(object sender, RoutedEventArgs e)
        {
            Button source = (Button)e.Source;
            folderDialog.ShowDialog();
            var FolderPath = folderDialog.SelectedPath;

            if (source.Name == "FirstFolderButton")
            {
                FFolderPath = FolderPath;
                GetImages(FolderPath, FirstList);
                if (FirstList.Items.Count > 0)
                {
                    FirstDeleteButton.IsEnabled = true;
                }
            }
            else if (source.Name == "SecondFolderButton")
            {
                SFolderPath = FolderPath;
                GetImages(FolderPath, SecondList);
                if (SecondList.Items.Count > 0)
                {
                    SecondDeleteButton.IsEnabled = true;
                }
            }
        }
        private void GetImages(string path, ListBox list)
        {
            list.Items.Clear();
            var imagePaths = Directory.GetFiles(path).Where(path => path.EndsWith(".png"));

            foreach (var imgpath in imagePaths)
            {
                StackPanel panel = new StackPanel();
                panel.Orientation = Orientation.Horizontal;

                ImageControl image = new()
                {
                    Source = new BitmapImage(new Uri(imgpath)),
                    Width = 80
                };
                panel.Children.Add(image);
                TextBlock textBlock = new TextBlock();

                textBlock.Text = System.IO.Path.GetFileName(imgpath);
                panel.Children.Add(textBlock);

                TextBlock HiddenPath = new TextBlock();
                HiddenPath.Text = imgpath;
                HiddenPath.Width = 0;
                panel.Children.Add(HiddenPath); 

                list.Items.Add(panel);
            }
        }

        public async void Calculate(object sender, RoutedEventArgs e)
        {
            CalculationProgressBar.Value = 0;
            CalculateButton.IsEnabled = false;
            CancellationButton.IsEnabled = true;
            cts = new CancellationTokenSource();

            var firstSelectedItem = (StackPanel)FirstList.SelectedItem;
            var secondSelectedItem = (StackPanel)SecondList.SelectedItem;

            var path1 = firstSelectedItem.Children.OfType<TextBlock>().Last().Text;
            var path2 = secondSelectedItem.Children.OfType<TextBlock>().Last().Text;


            var face1 = await File.ReadAllBytesAsync(path1, cts.Token);
            var face2 = await File.ReadAllBytesAsync(path2, cts.Token);

            var Embeding1 = await AddImageToDb(face1, path1);
            var Embeding2 = await AddImageToDb(face2, path2);

           


            CalculationProgressBar.Value += 50;
            var results = arcface.EmbRedyGetDistanceNSimilarity(Embeding1, Embeding2, cts.Token);

            if (!cts.Token.IsCancellationRequested)
            {
                var ListRes = results.ToList();
                SimilarityBlock.Text = ListRes[1].Item2.ToString();
                DistanceBlock.Text = ListRes[0].Item2.ToString();
                CalculationProgressBar.Value += 50;
            }
            else
            {
                CancellationRequested();
            }
            CalculateButton.IsEnabled = true;
            CancellationButton.IsEnabled = false;
        }

        public void Cancellation(object sender, RoutedEventArgs e)
        {
            if (cts != null)
            {
                cts.Cancel();
            }
            CancellationRequested();
        }
        public void CancellationRequested()
        {
            CalculationProgressBar.Value = 0;
            SimilarityBlock.Text = "null";
            DistanceBlock.Text = "null";
        }

        private void List_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (FirstList.SelectedItems.Count != 0 && SecondList.SelectedItems.Count != 0)
            {
                CalculateButton.IsEnabled = true;
            }
        }



        private async Task<float[]> AddImageToDb(byte[] img, string path)
        {
            var imageStream = new MemoryStream(img);
            
            var face = await Image.LoadAsync<Rgb24>(imageStream, cts.Token);
            float[] Emb;
            
            int hash = GetHash.GetHashCode(img);
            using (ImageContext db = new ImageContext())
            {
                List<ImageInfo> HashEq;
                lock (obj)
                {
                    HashEq = db.Photos.Where(image => image.Hash == hash).Take(1).Include(item => item.Value).ToList();
                }
                if (HashEq.Count == 1)
                    {
                    if (Enumerable.SequenceEqual(HashEq[0].Value.Image, img))
                        {
                            return Emb = BytesToFloats(HashEq[0].Embeddings);                  
                        }
                    }

                    Emb = await arcface.GetEmbedingsOutAsync(img, cts.Token);
                    ImageInfo imageInfo = new ImageInfo();

                    imageInfo.Hash = hash;
                    imageInfo.Embeddings = FloatsToBytes(Emb);
                    imageInfo.Path = path;
                    imageInfo.Value = new ImageValue { Image = img };
                    imageInfo.Name = System.IO.Path.GetFileName(path);
                lock (obj)
                {
                    db.Photos.Add(imageInfo);
                    db.SaveChanges();
                    return Emb;
                }
            }   
        }

        private void ClearDb(object sender, RoutedEventArgs e)
        {
            lock (obj)
            {
                using (ImageContext db = new ImageContext())
                {
                    foreach (var image in db.Photos)
                    {
                        db.Remove(image);
                    }
                    db.SaveChanges();
                }
            }
        }

        private void DeleteFirst(object sender, RoutedEventArgs e)
        {
            DeleteFromList((StackPanel)FirstList.SelectedItem);
            FirstList.Items.Remove(FirstList.SelectedItem);
        }
        private void DeleteSecond(object sender, RoutedEventArgs e)
        {
            DeleteFromList((StackPanel)SecondList.SelectedItem);
            SecondList.Items.Remove(SecondList.SelectedItem);   
        }
        private void DeleteFromList(StackPanel SelectedItem)
        {
            using (var db = new ImageContext())
            {
                var path = SelectedItem.Children.OfType<TextBlock>().Last().Text;
                byte[] img = File.ReadAllBytes(path);
                int hash = GetHash.GetHashCode(img);
                List<ImageInfo> HashEq;
                lock (obj)
                {
                    HashEq = db.Photos.Where(image => image.Hash == hash).Take(1).Include(item => item.Value).ToList();
                    if (HashEq.Count == 1)
                    {
                        if (Enumerable.SequenceEqual(HashEq[0].Value.Image, img))
                        {
                            db.Remove(HashEq[0]);
                            db.SaveChanges();
                        }
                    }
                }
            }
            SelectedItem.Children.Clear();
        }

        private void ShowDb(object sender, RoutedEventArgs e)
        {
            lock (obj)
            {
                using (var db = new ImageContext())
                {
                    foreach (var image in db.Photos)
                    {
                        bool clone = false;
                        foreach (StackPanel element in FirstList.Items)
                        {
                            string text = ((TextBlock)element.Children[1]).Text;
                            if (text == image.Name) { clone = true; break; }
                        }
                        if (!clone)
                        {
                            FirstList.Items.Add(BuildListElement(image));
                        }
                        

                        clone = false;
                        foreach (StackPanel element in SecondList.Items)
                        {
                            string text = ((TextBlock)element.Children[1]).Text;
                            if (text == image.Name) { clone = true; break; }
                        }
                        if (!clone)
                        {
                            SecondList.Items.Add(BuildListElement(image));
                        }
                    }
                }
            }
            if (FirstList.Items.Count > 0)
            {
                FirstDeleteButton.IsEnabled = true;
            }
            if (SecondList.Items.Count > 0)
            {
                SecondDeleteButton.IsEnabled = true;
            }
        }

        public StackPanel BuildListElement(ImageInfo image)
        {
            StackPanel panel = new StackPanel();
            panel.Orientation = Orientation.Horizontal;

            ImageControl webImage = new()
            {
                Source = new BitmapImage(new Uri(image.Path)),
                Width = 80
            };
            panel.Children.Add(webImage);
            TextBlock textBlock = new TextBlock();

            textBlock.Text = image.Name;
            panel.Children.Add(textBlock);

            TextBlock HiddenPath = new TextBlock();
            HiddenPath.Text = image.Path;
            HiddenPath.Width = 0;
            panel.Children.Add(HiddenPath); 
            
            return panel;

        }
        private byte[] FloatsToBytes(float[] array)
        {
            var byteBuffer = new byte[array.Length * 4];
            Buffer.BlockCopy(array, 0, byteBuffer, 0, byteBuffer.Length);
            return byteBuffer;
        }

        private float[] BytesToFloats(byte[] bytes)
        {
            var floatBuffer = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floatBuffer, 0, bytes.Length);
            return floatBuffer;
        }
    }

}