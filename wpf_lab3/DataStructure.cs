using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.Windows.Input;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.EntityFrameworkCore.Sqlite;


namespace Lab2vb1
{


    public class ImageInfo
    {
        [Key]
        public int IDPhoto { get; set; }
        public string Name { get; set; }
        public string Path { get; set; }
        public int Hash { get; set; }
        public byte[] Embeddings { get; set; }
        public ImageValue Value { get; set; }
    }

    public class ImageValue
    {
        [Key]
        [ForeignKey("ImageInfo")]
        public int IDPhoto { get; set; }
        public byte[] Image { get; set; }

    }

    public class ImageContext : DbContext
    {
        public DbSet<ImageInfo> Photos { get; set; }
        public DbSet<ImageValue> Details { get; set; }

        public ImageContext() => Database.EnsureCreated();
        protected override void OnConfiguring(Microsoft.EntityFrameworkCore.DbContextOptionsBuilder o) =>
            o.UseSqlite("Data Source=images.db");
    }

    public class GetHash
    {
        public static int GetHashCode(byte[] data)
        {
            const int p = 13252748;
            int hash;
            hash = 1382451931;

            for (int i = 0; i < data.Length; i++)
                hash = (hash ^ data[i]) * p;

            hash += hash << 11;
            hash ^= hash >> 5;
            hash += hash << 6;
            hash ^= hash >> 10;
            hash += hash << 9;
            return hash;
        }
    }

}
